import os
import numpy as np
import torch
from torch.distributions.normal import Normal

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
        self.kart = all_players[np.random.choice(len(all_players))]
        self.agent = load_model(os.path.relpath('action'))
        self.vision = load_vision_model(os.path.relpath('vision'))
        
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
        img = torch.tensor(image, dtype=torch.float).permute(2,0,1)[None]
        # print(img.shape)
        # heatmap = self.vision(img)
        # decision = self.agent(torch.sigmoid(heatmap)).detach().numpy()[0]
        row = self.agent(img)[0]#.detach().numpy()[0]
        # print(row)
        # steer, acceleration, brake = decision
        steer_dist = Normal(row[0],torch.abs(row[1])+0.001) #make sure sigma is positive and non-zero
        acc_dist = Normal(row[2],torch.abs(row[3])+0.001) #make sure sigma is positive and non-zero
        brake_dist = Normal(row[4],torch.abs(row[5])+0.001) #make sure sigma is positive and non-zero
        steer = steer_dist.sample()
        acceleration = acc_dist.sample()
        brake = brake_dist.sample()
        print(steer,  acceleration, brake)
        action['steer'] = steer
        action['acceleration'] = acceleration
        action['brake'] = False if brake < 0.5 else True

        return action

if __name__ == '__main__':
    players = HockeyPlayer()