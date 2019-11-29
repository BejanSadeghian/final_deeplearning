import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np
import pystk
from agent_0.player import HockeyPlayer
from agent_0.model import Action, save_model
from agent_0.vision_model import Vision, load_vision_model
import itertools
from torch.distributions.normal import Normal
# from utils import load_data

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)

def getRMSE(list_preds, list_targets, idx):
    predicted = np.array([x[idx] for x in list_preds])
    targets = np.array([x[idx] for x in list_targets])
    return np.sqrt(((predicted - targets)**2).mean())


def rollout_agent(device, vision, action, n_steps=200):
    race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)

    k = pystk.Race(race_config)
    k.start()
    for i in range(5): #Skip the first 5 steps since its the game starting
        k.step() 
    try:
        data = []
        for n in range(n_steps):
            img = torch.tensor(np.array(k.render_data[0].image), dtype=torch.float).to(device).permute(2,0,1)
            # heatmap = vision(img)
            # p = action(torch.sigmoid(heatmap))[0]
            p = action(img)[0]
            # print(p[0])
            k.step(pystk.Action(steer=float(p[0]), acceleration=float(p[1]), brake=float(p[2])>0.5)) #TODO: remove /10 later
            # print(pystk.Action(acceleration=float(p[0]), steer=float(p[1]), brake=float(p[2])>0.5))
            la = k.last_action[0]
            # print((la.acceleration, la.steer, la.brake))
            # print('end')
            data.append((np.array(k.render_data[0].image), (la.steer, la.acceleration, la.brake))) #TODO: remove /10 later
    finally:
        k.stop()
        del k
    return data

def rollout(device, vision, action, n_steps=200):
    race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
    o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.AI_CONTROL, team = 0)
    race_config.players.append(o)
    k = pystk.Race(race_config)
    
    k.start()
    for i in range(5): #Skip the first 5 steps since its the game starting
        k.step() 
    try:
        data = []
        for n in range(n_steps):
            k.step()
            la = k.last_action[0]

            data.append((np.array(k.render_data[0].image), (la.steer, la.acceleration, la.brake))) #TODO: remove /10 later 3 in this script and 1 in player
    finally:
        k.stop()
        del k
    return data

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Action(normalize=True, inference=False).to(device)
    model.train(True)
    vision_model = load_vision_model('vision') #Vision().to(device)
    vision_model.to(device)
    vision_model.train(False)
    vision_model.eval()

    loss = torch.nn.L1Loss()
    loss.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    batch_size = args.batch_size
    max_steps = args.max_steps
    num_players = 1

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)  
    
    #PySTK init
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)

    train_data = list(itertools.chain(*[rollout(device, vision_model, model) for it in range(10)]))
    global_step = 0
    for e in range(args.epochs):
        all_targets = []
        all_predictions = []

        # state = pystk.WorldState()
        # print('a')

        if e > 1:
            train_data.extend(rollout_agent(device, vision_model, model))
        
        np.random.shuffle(train_data)
        # print(train_data)
        for iteration in range(0, len(train_data)-batch_size+1, batch_size):
            # print('\rEpoch: {} Step: {} of {}'.format(e,iteration//batch_size,batch_size), end='\r')
            print(iteration)

            batch_data = torch.as_tensor([train_data[i][0] for i in range(iteration, iteration+batch_size)]).permute(0,3,1,2).float()
            batch_label = torch.as_tensor([train_data[i][1] for i in range(iteration, iteration+batch_size)]).float()

            # heatmap = vision_model(batch_data.to(device))
            # p = model(torch.sigmoid(heatmap))
            p = model(batch_data.to(device))
            
            samples = []
            log_probs = []
            for row in p:
                steer_dist = Normal(row[0],torch.abs(row[1])+0.001) #make sure sigma is positive and non-zero
                acc_dist = Normal(row[2],torch.abs(row[3])+0.001) #make sure sigma is positive and non-zero
                brake_dist = Normal(row[4],torch.abs(row[5])+0.001) #make sure sigma is positive and non-zero
                steer = steer_dist.sample()
                acc = acc_dist.sample()
                brake = brake_dist.sample()
                s = torch.tensor([steer, acc, brake], dtype=torch.float, requires_grad=True)
                samples.append(s)
                lp = torch.tensor([
                    -steer_dist.log_prob(steer),
                    -acc_dist.log_prob(acc),
                    -brake_dist.log_prob(brake)
                ], requires_grad=True)
                log_probs.append(lp)
            samples = torch.stack(samples)
            log_probs = torch.stack(log_probs)
            # print(samples)
            # print(batch_label.to(device))
            # print(log_probs)
            # l = loss(samples, batch_label.to(device))
            # print(torch.abs(samples - batch_label.to(device)))
            # input()
            l = (log_probs.cpu() * torch.abs(samples.cpu() - batch_label.cpu())).sum()
            print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            all_targets.append(batch_label.detach().cpu().numpy())
            all_predictions.append(p.squeeze().detach().cpu().numpy())
            if args.logdir is not None:
                train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1

        # if args.logdir is not None:
        #     train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        #     train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        #     train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        save_model(model, 'action')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()
    train(args)