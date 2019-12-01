import torch

import os
import numpy as np
import pystk
import PIL.Image

def rollout(epoch, max_roll=1000, skip=5, n_players=4, save='vision_data_valid'):
    race_config = pystk.RaceConfig(track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
    race_config.players.pop()

    for n in range(n_players):
        o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.AI_CONTROL, team = n % 2)
        race_config.players.append(o)

    k = pystk.Race(race_config)

    k.start()
    for _ in range(2): #Skip the first x steps since its the game starting
        k.step()
    
    try:
        for step in range(0, max_roll):
            print(step)
            for driver in range(n_players):
                img_rgb = np.array(k.render_data[driver].image)
                img_types = np.array(np.array(k.render_data[driver].instance) >> pystk.object_type_shift, dtype=int)

                if save is not None:
                    PIL.Image.fromarray(img_rgb).save(os.path.join(save, 'player_e{}_s{}_d{}.png'.format(epoch, step, driver)))
                    np.savetxt(os.path.join(save, 'player_e{}_s{}_d{}.txt').format(epoch, step, driver), img_types, fmt='%d')

            for _ in range(skip):
                k.step()
    finally:
        k.stop()
        del k

if __name__ == '__main__':
    #PySTK init
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)

    for e in range(1):
        rollout(e)