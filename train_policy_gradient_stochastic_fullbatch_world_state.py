import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np
import pystk
from agent_policy_gradient_stochastic.player import HockeyPlayer
from agent_policy_gradient_stochastic.model import Action, save_model
from agent_policy_gradient_stochastic.vision_model import Vision, load_vision_model
import itertools
from torch.distributions.normal import Normal
from torchvision import transforms
from PIL import Image


class position_loss(torch.nn.Module):

    def __init__(self, goal_position, goal_weight=1):
        super().__init__()
        self.goal_position = torch.tensor((0, goal_position), dtype=torch.float)
        self.goal_weight = goal_weight
    
    def forward(self, kart:torch.tensor, puck:torch.tensor, log_prob:torch.tensor):
        #MSE
        if len(kart.shape) == 3:
            for i in range(kart.shape[0]):
                print(kart.shape[0])
                
                p = puck[i].clone().detach()
                k = kart[i].clone().detach()
                diff = k.squeeze(0) - p.squeeze(0)
                diff_sq = (diff)**2

                reward = (diff_sq).sum(1).sqrt() + \
                    self.goal_weight * ((self.goal_position - p.squeeze(0))**2).sum(1).sqrt()
                    #  + \
                    # 100 * (90 - ((180/3.14) * torch.atan((diff_sq).sqrt()[:,1] / (diff_sq).sqrt()[:,0])))**2

                # new_delta = (reward * log_prob[i].squeeze(0) * time).mean()
                # print('reward',log_prob)
                if i == 0:
                    time = torch.tensor(i)[None]
                    all_rewards = reward.clone()[None]
                    all_logprob = log_prob.clone()[None]

                    r = -reward.clone()[None] 
                    lp = -log_prob.clone()[None]
                    weighted_reward = (lp * (r - r.mean())).sum()
                    all_weighted_rewards = weighted_reward.clone()[None]
                    # delta = new_delta.clone()[None]
                else:
                    time = torch.cat((time, torch.tensor(i)[None]),0)
                    all_rewards = -1*torch.cat((all_rewards, reward[None]),0) #flipped to a maximization problem
                    all_logprob = torch.cat((all_logprob, log_prob[None]),0)
                    r = -reward.clone()[None] 
                    lp = -log_prob.clone()[None]
                    weighted_reward = (lp * (r - r.mean())).sum()
                    all_weighted_rewards = torch.cat((all_weighted_rewards, weighted_reward[None]),0)
                    # delta = torch.cat((delta, new_delta[None]),0)
                print(1,all_weighted_rewards)
            # delta = -all_logprob * (all_rewards - all_rewards.mean()) #flipped to a maximization problem
            delta = all_weighted_rewards.mean()
            result = delta.squeeze(0)
            # print(delta.shape)
        else:
            time = torch.tensor(range(puck.shape), dtype=torch.float).sqrt()
            p = puck.clone().detach()
            k = kart.clone().detach()
            delta = ((k - p)**2).sum(1) + self.goal_weight * ((self.goal_position - p)**2).sum(1)
            delta = delta*log_prob*time
            result = delta.mean()
        return result


class visual_loss(torch.nn.Module):

    def __init__(self, image_dims=(100,130)):
        super().__init__()
        self.image_dims = image_dims

    def _get_reward(self, input, target, team_no):
        dim_c, dim_y, dim_x = input.shape
        b = dim_x // 2
        weights = torch.tensor([x + b if x < 0 else -x + b for x in np.linspace(-b, b, dim_x)], dtype=torch.float).detach()

        if torch.sigmoid(input[0]).sum() > (1/6) * self.image_dims[0] * self.image_dims[1]:
            input_reward = torch.tensor(0.0, dtype=torch.float)
        else:
            input_reward = torch.sigmoid(input[0]).mul(weights).sum() + torch.sigmoid(input[0]).mul(weights).mul(input[team_no]).sum()
        if torch.sigmoid(target[0]).sum() > (1/6) * self.image_dims[0] * self.image_dims[1]:
            target_reward = torch.tensor(0.0, dtype=torch.float)
        else:
            target_reward = torch.sigmoid(target[0]).mul(weights).sum() + torch.sigmoid(target[0]).mul(weights).mul(target[team_no]).sum()
        return input_reward, target_reward

    def forward(self, input:torch.tensor, target:torch.tensor=None, team_no:int=1):
        """
        Looking to reward the frames with the puck included and additionally reward frames 
            that contain the puck infront of the goal.

        Loss = ( sum(puck pixels in frame prior) + sum(puck pixels * goal pixels in frame prior) ) 
                - ( sum(puck pixels in frame after) + sum(puck pixels * goal pixels in frame after) )

        Expects class 0 to be the puck and 1,2 to be the goals for team 1 and 2 respectively
        Inputs:
            :input: <tensor> (3,H,W) or (B,3,H,W) heatmap of response image BEFORE taking action (after sigmoid transform)
            :target: <tensor> (3,H,W) heatmap of response image AFTER taking action (after sigmoid transform) -- if input has a batch dim then this is  not used
            :team_no: <int> team 1 or team 2 (dependent on what goal you see at start)

        Output:
            :loss: <float> the calculated loss
        """
        if len(input.shape) == 4:
            log_prob = target[0]
            input_image = input[0].squeeze()
            target_image = input[1].squeeze()
            input_reward, target_reward = self._get_reward(input_image, target_image, team_no)
            input_reward = input_reward
            target_reward = target_reward
            reward_delta = (input_reward - target_reward if (input_reward - target_reward) != 0 else 0.001) * log_prob
            for i in range(2,input.shape[0]):
                log_prob = target[i-1]
                input_image = input[i-1].squeeze()
                target_image = input[i].squeeze()
                input_reward, target_reward = self._get_reward(input_image, target_image, team_no)
                # input_reward += input_reward_new
                # target_reward += target_reward_new
                reward_delta = (input_reward - target_reward if (input_reward - target_reward) != 0 else 0.001) * log_prob
        else:
            print('instance')
            input_reward, target_reward = self._get_reward(input, target, team_no)
            reward_delta = input_reward - target_reward
        return reward_delta

# def sample_action(p):
#     steer_dist = Normal(p[0],torch.abs(p[1])+0.001) #make sure sigma is positive and non-zero
#     acc_dist = Normal(p[2],torch.abs(p[3])+0.001) #make sure sigma is positive and non-zero
#     brake_dist = Normal(p[4],torch.abs(p[5])+0.001) #make sure sigma is positive and non-zero

#     steer = steer_dist.sample()
#     acc = acc_dist.sample()
#     brake = brake_dist.sample()

#     actions = torch.tensor([steer, acc, brake], dtype=torch.float, requires_grad=True)
#     log_prob = (steer_dist.log_prob(steer) + acc_dist.log_prob(acc) + brake_dist.log_prob(brake))
#     # print('prob',log_prob)
#     return actions, log_prob

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)

    #Initialize or Load Models
    action_model = Action(normalize=True, inference=False).to(device) #No inference because we do not want to resize images (vision model does that for us)
    action_model.train(True)
    vision_model = load_vision_model('vision') #Vision().to(device)
    vision_model.to(device)
    vision_model.train(False)
    vision_model.eval()

    #Define transform, loss, and optimizer
    n_steps = args.n_steps #300
    batch_size = args.batch_size
    image_to_tensor = transforms.ToTensor()

    # loss = visual_loss()
    # loss = torch.nn.MSELoss()
    goal_position = 65
    loss = position_loss(goal_position, 1)
    optimizer = torch.optim.Adam(action_model.parameters(), lr=args.learning_rate)

    train_logger = None
    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)  
    
    #PySTK init
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)

    #Start Train Loop
    global_step = 0
    allowed_batchs = 0
    for e in range(args.epochs):
        print('Epoch:',e)
        if e % 20 == 0:
            allowed_batchs += 10
        
        for b in range(batch_size):
            race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
            race_config.players.pop()
            o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart='hexley', team = 0)
            race_config.players.append(o)
            # print(race_config.players.pop())
            print('batch',b)
            image_to_tensor = transforms.ToTensor()
            k = pystk.Race(race_config)
            state = pystk.WorldState()
            k.start()
            for i in range(10): #Skip the first 10 steps since its the game starting
                k.step() 
                state.update()
            # print(dir(state))
            # print(1,state.karts[0].location)
            # print(2,state.soccer.ball.location)
            # print(3,state.soccer.goal_line)
            # print(4,dir(state.soccer.score))
            # input()
            try:
                all_losses = []
                last_image = np.array(k.render_data[0].image)
                img = image_to_tensor(last_image)
                heatmap, resized_image = vision_model(img.to(device))
                restart = True
                for n in range(1, n_steps+1):
                    input_heatmap = heatmap.clone() #heatmap from past step, no longer detached for batch processing
                    resized_image = resized_image #no longer detached for batch processing
                    puckmap = torch.tensor(vision_model.find_puck(input_heatmap[0]), dtype=torch.float).unsqueeze(0)
                    input_heatmap[:,0] = puckmap * 10
                    

                    #Save image
                    if n % 10 == 0: #Just to see what is going on
                        save = 'policy_training'
                        if save is not None:
                            if not os.path.exists(save):
                                os.makedirs(save)
                            image_orig = torch.tensor(last_image, dtype=torch.float).permute(2,0,1)[None]
                            Image.fromarray(np.uint8(torch.sigmoid(input_heatmap.squeeze(0).permute(1,2,0)).detach().numpy()*255)).save(os.path.join(save, 'player{}-{}a.png'.format(e,n)))
                            Image.fromarray(np.uint8(image_orig.squeeze(0).permute(1,2,0).detach().numpy())).save(os.path.join(save, 'player{}-{}b.png'.format(e,n)))
            
                    #Action model
                    heatmap_transformed = torch.sigmoid(input_heatmap)
                    combined_image = torch.cat((resized_image, heatmap_transformed), 1) #resized_image from past step (Add detach here?)
                    p = action_model(combined_image)[0]
                    actions, log_probs = HockeyPlayer.sample_action(p) #Changed to use the static method
                    if restart:
                        kart = torch.tensor((state.karts[0].location[0],state.karts[0].location[2]), dtype=torch.float)[None]
                        puck = torch.tensor((state.soccer.ball.location[0],state.soccer.ball.location[2]), dtype=torch.float)[None]
                        log_probabilities = log_probs[None]
                        restart = False
                    else:
                        kart = torch.cat((kart, torch.tensor((state.karts[0].location[0],state.karts[0].location[2]), dtype=torch.float)[None]))
                        puck = torch.cat((puck, torch.tensor((state.soccer.ball.location[0],state.soccer.ball.location[2]), dtype=torch.float)[None]))
                        log_probabilities = torch.cat((log_probabilities, log_probs[None]))
                    # la = k.last_action[0]
                    # k.step(pystk.Action(steer=float(la.steer), acceleration=float(la.acceleration), brake=float(la.brake)>0.5))
                    k.step(pystk.Action(steer=float(actions[0]), acceleration=float(actions[1]), brake=float(actions[2])>0.5))
                    state.update()
                    if n == 100:
                        print(float(actions[0]), float(actions[1]), float(actions[2])>0.5)
                        print(p)

                    #Get updated image to calculate reward and use for next step
                    last_image = np.array(k.render_data[0].image)
                    img = image_to_tensor(last_image)
                    heatmap, resized_image = vision_model(img.to(device))

            finally:
                k.stop()
                del k
            if b == 0:
                karts = kart[None]
                pucks = puck[None]
                log_probabilities_set = log_probabilities[None]
            else:
                karts = torch.cat((karts, kart[None]),0)
                pucks = torch.cat((pucks, puck[None]),0)
                log_probabilities_set = torch.cat((log_probabilities_set, log_probabilities[None]),0)
        # print(karts.shape, pucks.shape, log_probabilities_set.shape)
        #Calculate loss and take step
        l = loss(karts[1:], pucks[1:], log_probabilities_set[1:])
        print('loss',l)
        # print('allowed batchs',allowed_batchs)
        optimizer.zero_grad()
        l.backward() #retain_graph=True
        optimizer.step()

        #Record Loss
        all_losses.append(l.cpu().detach().numpy())
        if train_logger is not None:
            train_logger.add_scalar('loss', l.cpu(), global_step=global_step) 
        global_step += 1
        restart = True

        # all_losses, global_step = one_trajectory_run(device, vision_model, action_model, loss, optimizer, train_logger=train_logger, n_steps=200, global_step=global_step)
        if train_logger is not None:
            train_logger.add_scalar('Trajectory_Avg_Loss', np.mean(all_losses), global_step=e)

        save_model(action_model, 'action_stochastic_batch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)