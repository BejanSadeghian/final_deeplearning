import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np
import pystk
from agent_policy_gradient.player import HockeyPlayer
from agent_policy_gradient.model import Action, save_model
from agent_policy_gradient.vision_model import Vision, load_vision_model
import itertools
# from torch.distributions.normal import Normal
from torchvision import transforms
from PIL import Image


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
            input_image = input[0].squeeze()
            target_image = input[1].squeeze()
            input_reward, target_reward = self._get_reward(input_image, target_image, team_no)
            input_reward = input_reward
            target_reward = target_reward
            for i in range(2,input.shape[0]):
                input_image = input[i-1].squeeze()
                target_image = input[i].squeeze()
                input_reward_new, target_reward_new = self._get_reward(input_image, target_image, team_no)
                input_reward += input_reward_new
                target_reward += target_reward_new
                # input_rewards = torch.cat((input_reward, input_reward_new[None]))
                # target_reward = torch.cat((target_reward, target_reward_new[None]))
            # input_reward = input_reward.sum()
            # target_reward = target_reward.sum()
        else:
            print('instance')
            input_reward, target_reward = self._get_reward(input, target, team_no)
        
        return input_reward - target_reward if (input_reward - target_reward) != 0 else -0.001

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
    image_to_tensor = transforms.ToTensor()

    loss = visual_loss()
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
    for e in range(args.epochs):
        print('Epoch:',e)

        race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
        race_config.players.pop()
        o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart='hexley', team = 0)
        race_config.players.append(o)
        # print(race_config.players.pop())

        image_to_tensor = transforms.ToTensor()
        k = pystk.Race(race_config)
        k.start()
        for i in range(10): #Skip the first 10 steps since its the game starting
            k.step() 

        try:
            team_no = 1
            all_losses = []
            last_image = np.array(k.render_data[0].image)
            img = image_to_tensor(last_image)
            heatmap, resized_image = vision_model(img.to(device))
            for n in range(1, n_steps):
                input_heatmap = heatmap.clone() #heatmap from past step, no longer detached for batch processing
                resized_image = resized_image #no longer detached for batch processing
                puckmap = torch.tensor(vision_model.find_puck(input_heatmap[0]), dtype=torch.float).unsqueeze(0)
                input_heatmap[:,0] = puckmap * 10
                if n == 1:
                    heatmaps = input_heatmap.clone()
                else:
                    new_heatmap = input_heatmap.clone()
                    heatmaps = torch.cat((heatmaps, new_heatmap))

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
                k.step(pystk.Action(steer=float(p[0]), acceleration=float(p[1]), brake=float(p[2])>0.5))
                if n == 100:
                    print(float(p[0]), float(p[1]), float(p[2])>0.5)
                # la = k.last_action[0]

                #Get updated image to calculate reward and use for next step
                last_image = np.array(k.render_data[0].image)
                img = image_to_tensor(last_image)
                heatmap, resized_image = vision_model(img.to(device))

            #Calculate loss and take step
            l = loss(input=heatmaps, team_no=team_no)

            optimizer.zero_grad()
            l.backward() #retain_graph=True
            optimizer.step()

            #Record Loss
            all_losses.append(l.cpu().detach().numpy())
            if train_logger is not None:
                train_logger.add_scalar('loss', l.cpu(), global_step=global_step) 
            global_step += 1
        finally:
            k.stop()
            del k

        # all_losses, global_step = one_trajectory_run(device, vision_model, action_model, loss, optimizer, train_logger=train_logger, n_steps=200, global_step=global_step)
        if train_logger is not None:
            train_logger.add_scalar('Trajectory_Avg_Loss', np.mean(all_losses), global_step=e)

        save_model(action_model, 'action_deterministic_no_batch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)