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

if __name__ == '__main__':
    for e in range(100):
        o = rollout()
    print(o[0])
    