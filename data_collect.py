import torch
from torch import nn
from torch import optim
import pystk
import numpy as np
import PIL.Image
import os, sys
import csv


config = pystk.GraphicsConfig.hd()
config.screen_width = 400
config.screen_height = 300
pystk.init(config)

def rollout(epoch=0, num_players=2, max_steps=1000, track='icy_soccer_field', save='train_data_two_players'):
    race_config = pystk.RaceConfig(num_kart=num_players, track=track, mode=pystk.RaceConfig.RaceMode.SOCCER, difficulty=2)
    race_config.players.pop()

    if not os.path.exists(save):
        os.makedirs(save)

    for i in range(num_players):
        o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.AI_CONTROL, team = int(i%2.0))
        race_config.players.append(o)

    active_players = race_config.players
    print('Number of Players',len(active_players))

    k = pystk.Race(race_config)
    k.start()
    try:
        k.step()

        # all_images = []
        all_actions = []
        state = pystk.WorldState()
        state.update()
        # print(dir(state.players[0].kart))
        # input()
        for t in range(max_steps):
            # print('step:',t)
            state.update()

            s = k.step()
            t_actions = []
            for i in range(num_players):
                la = k.last_action[i]
                img = np.array(k.render_data[i].image)
                a = [img, la.steer, la.acceleration, la.brake]
                a.extend(state.players[0].kart.rotation)
                a.extend(state.players[0].kart.velocity)
                a.extend(state.players[0].kart.location)
                
                t_actions.append(a)
                if save is not None:
                    PIL.Image.fromarray(img).save(os.path.join(save, 'player_%05d_%02d_%05d.png' % (epoch, i, t)))
                    with open(os.path.join(save, 'player_%05d_%02d_%05d.csv' % (epoch, i, t)), mode='w') as new_data:
                        writer = csv.writer(new_data)
                        writer.writerow(a[1:])
            
            all_actions.append(t_actions)
            if not s:
                break
        print('epoch',e,'score',state.soccer.score)
    finally:
        k.stop()
        del k
    
    return all_actions


# class Tournament:
#     _singleton = None

#     def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
#         assert Tournament._singleton is None, "Cannot create more than one Tournament object"
#         Tournament._singleton = self

#         self.graphics_config = pystk.GraphicsConfig.hd()
#         self.graphics_config.screen_width = screen_width
#         self.graphics_config.screen_height = screen_height
#         pystk.init(self.graphics_config)

#         self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
#         self.race_config.players.pop()
        
#         self.active_players = []
#         for p in players:
#             if p is not None:
#                 self.race_config.players.append(p.config)
#                 self.active_players.append(p)
                
        
#         self.k = pystk.Race(self.race_config)

#         self.k.start()
#         self.k.step()

#     def play(self, save=None, max_frames=50):
#         state = pystk.WorldState()
#         if save is not None:
#             import PIL.Image
#             import os
#             if not os.path.exists(save):
#                 os.makedirs(save)

#         for t in range(max_frames):
#             print('\rframe %d' % t, end='\r')

#             state.update()

#             list_actions = []
#             for i, p in enumerate(self.active_players):
#                 player = state.players[i]
#                 image = np.array(self.k.render_data[i].image)
                
#                 action = pystk.Action()
#                 player_action = p(image, player)
#                 for a in player_action:
#                     setattr(action, a, player_action[a])
                
#                 list_actions.append(action)

#                 if save is not None:
#                     PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))

#             s = self.k.step(list_actions)
#             if not s:  # Game over
#                 break

#         if save is not None:
#             import subprocess
#             for i, p in enumerate(self.active_players):
#                 dest = os.path.join(save, 'player%02d' % i)
#                 output = save + '_player%02d.mp4' % i
#                 subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
#         if hasattr(state, 'soccer'):
#             return state.soccer.score
#         return state.soccer_score

#     def close(self):
#         self.k.stop()
#         del self.k

if __name__ == '__main__':
    for e in range(100):
        o = rollout()
    print(o[0])
    