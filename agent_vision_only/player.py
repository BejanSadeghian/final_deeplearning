import os
import numpy as np
import torch
from torch.distributions.normal import Normal
from PIL import Image
from datetime import datetime
from torchvision import transforms
import torchvision

from .model import load_model, Action
from .vision_model import load_vision_model, Vision

class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        all_players = ['hexley']
        self.kart = all_players[np.random.choice(len(all_players))]

        #Load models and set to eval/no train mode
        self.agent = load_model(os.path.relpath('action_stochastic_batch'))
        self.agent.eval()
        self.agent.train(False)
        self.vision = load_vision_model(os.path.relpath('vision'))
        self.vision.eval()
        self.vision.train(False)

        self.counter = 0
        self.transformer = transforms.ToTensor()
    
    @staticmethod
    def sample_action(p):
        steer_dist = Normal(p[0],torch.abs(p[1])+0.001) #make sure sigma is positive and non-zero
        acc_dist = Normal(p[2],torch.abs(p[3])+0.001) #make sure sigma is positive and non-zero
        brake_dist = Normal(p[4],torch.abs(p[5])+0.001) #make sure sigma is positive and non-zero

        steer = steer_dist.sample()
        acc = acc_dist.sample()
        brake = brake_dist.sample()

        actions = torch.tensor([steer, acc, brake], dtype=torch.float, requires_grad=True)
        log_prob = (steer_dist.log_prob(steer) + acc_dist.log_prob(acc) + brake_dist.log_prob(brake))
        # print('prob',log_prob)
        return actions, log_prob
    
    @staticmethod
    def get_rotation(q):
        qw,qx,qy,qz = q
        # a = 2*(w*z + x*y)
        # b = 1-2*(y**2 + z**2)
        # angle = np.arctan2(a,b)

        heading = np.arctan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
        # attitude = np.arcsin(2*qx*qy + 2*qz*qw) 
        # bank = np.arctan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)

        return (heading * (180/np.pi))
        # return angle

    def to_multi_channel(self, heatmap, class_range=list(range(1,10)), classes = None):
        """
        heatmap shape (C,H,W)
        """    
        CLASSES = [1,8]

        def to_single_channel(heatmap):
            heatmap = heatmap.float()
            return ((torch.exp(heatmap.cpu()) / torch.exp(heatmap.cpu()).sum(0)).max(0).indices)

        ## For BCE Loss
        heatmap = heatmap.float()
        # print(np.unique(heatmap.detach().numpy()))
        if len(heatmap.shape) == 3:
            tgt = to_single_channel(heatmap)
        else:
            tgt = heatmap.clone()
        output_target = []
        tgt = tgt.detach().cpu().float().numpy()

        tgt_classes = tgt.copy()
        if classes is not None:
            for ix, c in enumerate(class_range):
                if c not in classes:
                    tgt[tgt == float(c)] = 0
            tgt[tgt != 0] = -1
            tgt += 1
            output_target.append(torch.tensor(tgt, dtype=torch.float)[None])
        else:
            classes = list(range(len(CLASSES)+1))

        for c in classes:
            tgt_temp = np.zeros(tgt_classes.shape)
            tgt_temp[tgt_classes == float(c)] = 1
            output_target.append(torch.tensor(tgt_temp, dtype=torch.float)[None])

        tgt = torch.cat(output_target)
        return tgt
        
    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3) #400 width and 300 height
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """
        # print((player_info.kart.location)) #if z < 0 at start you're facing the red goal (team 1) otherwise the blue 
        # print((player_info.kart.rotation))
        # angle = get_rotation(player_info.kart.rotation)

        img = self.transformer(image)
        image_orig = torch.tensor(image, dtype=torch.float).permute(2,0,1)[None]
        heatmap = self.vision(img)
        # heatmap = torch.sigmoid(heatmap)
        # puckmap = self.vision.find_puck(heatmap[0])
        # puckmap = torch.tensor(puckmap, dtype=torch.float).unsqueeze(0)
        # heatmap[:,0] = puckmap #* heatmap[:,0] # only looks at the deetected puck

        #Save Outputs
        save = 'agent_view_heatmap'
        if save is not None:
            if not os.path.exists(save):
                os.makedirs(save)
            heatmap = heatmap.squeeze(0)

            # Image.fromarray(np.uint8(heatmap.squeeze(0).permute(1,2,0).detach().numpy()*255)).save(os.path.join(save, 'player{}a.png'.format(self.counter)))
            Image.fromarray(np.uint8(image_orig.squeeze(0).permute(1,2,0).detach().numpy())).save(os.path.join(save, 'player{}a.png'.format(self.counter)))
            fp = os.path.join(save, 'player{}b.png'.format(self.counter))
            torchvision.utils.save_image(self.to_multi_channel(heatmap), fp)

            detect = self.vision.detect(heatmap)
            detection = torch.zeros(heatmap.shape)
            if detect[0]:
                _,y,x = detect
                detection[0,y-5:y+5, x-5:x+5] = 1
                fp = os.path.join(save, 'player{}c.png'.format(self.counter))
                torchvision.utils.save_image(detection, fp)

        self.counter += 1
        print('detect',)
        # # decision = self.agent(torch.sigmoid(heatmap)).detach().numpy()[0]
        # # print(resized_image.shape, heatmap.shape)
        # combined_image = torch.cat((resized_image, heatmap),1)
        # p = self.agent(combined_image)[0]#.detach().numpy()[0]
        # actions, log_probs = self.sample_action(p)
        # # print(row)
        # steer, acceleration, brake = actions
        # # steer_dist = Normal(row[0],torch.abs(row[1])+0.001) #make sure sigma is positive and non-zero
        # # acc_dist = Normal(row[2],torch.abs(row[3])+0.001) #make sure sigma is positive and non-zero
        # # brake_dist = Normal(row[4],torch.abs(row[5])+0.001) #make sure sigma is positive and non-zero
        # # steer = steer_dist.sample()
        # # acceleration = acc_dist.sample()
        # # brake = brake_dist.sample()
        # print(steer,  acceleration, brake)
        # action['steer'] = steer
        # action['acceleration'] = acceleration
        # action['brake'] = False if brake < 0.5 else True
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        return action

if __name__ == '__main__':
    players = HockeyPlayer()