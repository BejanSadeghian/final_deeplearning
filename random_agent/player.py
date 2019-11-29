import numpy as np


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
    
    @staticmethod
    def get_rotation(q):
        qw,qx,qy,qz = q
        # a = 2*(w*z + x*y)
        # b = 1-2*(y**2 + z**2)
        # angle = np.arctan2(a,b)

        heading = np.arctan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
        attitude = np.arcsin(2*qx*qy + 2*qz*qw) 
        bank = np.arctan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)

        print(heading * (180/np.pi), attitude, bank)
        # return angle

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        # print((player_info.kart.location))
        # print((player_info.kart.rotation))
        self.get_rotation(player_info.kart.rotation)
        action = {'acceleration': 0.5, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': -1}
        """
        Your code here.
        """

        return action

