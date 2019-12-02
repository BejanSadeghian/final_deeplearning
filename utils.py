import torch
import os
import csv
import numpy as np
import ast
from PIL import Image
from torchvision import transforms
import re

class AgentData(torch.utils.data.DataLoader):

    def __init__(self, dataset_path, norm=False, recalc_norm=False, resize=(130,100)):
        self.norm = norm
        self.dataset_path = dataset_path
        self.resize = resize

        self.ids = [x for x in os.listdir(self.dataset_path) if x.endswith('.csv')]
        data = []
        for i in self.ids:
            with open(os.path.join(self.dataset_path, i)) as file_obj:
                reader = csv.reader(file_obj, delimiter=',')
                for ix, row in enumerate(reader):
                    if ix != 0:
                        break
                    data.append((i[:-4]+'.png', np.array([ast.literal_eval(x) if x.lower() != 'false' and x.lower() != 'true' else bool(ast.literal_eval(x)) for x in row])))
        self.data = data
        if recalc_norm:
            norm_calc = []
            for d in data:
                norm_calc.append(np.array(Image.open(os.path.join(self.dataset_path, d[0]))))
            norm_calc = np.stack(norm_calc)
            print(norm_calc.shape)
            print(torch.tensor(norm_calc, dtype=torch.float).mean([0,1,2]))
            print(torch.tensor(norm_calc, dtype=torch.float).std([0,1,2]))
        self.mean = torch.tensor([8.9478, 8.9478, 8.9478], dtype=torch.float) 
        self.std = torch.tensor([47.0021, 42.1596, 39.2562], dtype=torch.float)
        # print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        targets = self.data[idx][1:]
        img = Image.open(os.path.join(self.dataset_path, image))
        # img = img.resize(self.resize) #Resize image
        if self.norm:
            image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        else:
            image_to_tensor = transforms.ToTensor()
        img = image_to_tensor(img)
        # print(img.shape)
        return (img, torch.tensor(targets, dtype=torch.float))

class ClassifierData(torch.utils.data.DataLoader):

    def __init__(self, dataset_path, norm=False, recalc_norm=False, resize=(130,100), classes = [15,17,18]):
        #Norm handled in model
        self.norm = norm
        self.dataset_path = dataset_path
        self.resize = resize
        self.classes = classes

        self.ids = [x for x in os.listdir(self.dataset_path) if x.endswith('.csv') and bool(re.search('player00',x, flags=re.I))]
        self.ids.sort()
        data = []
        for i in self.ids:
            with open(os.path.join(self.dataset_path, i)) as file_obj:
                reader = csv.reader(file_obj, delimiter=',')
                for ix, row in enumerate(reader):
                    # if ix != 0:
                    #     break
                    data.append((i[:-4], np.array([ast.literal_eval(x) if x.lower() != 'false' and x.lower() != 'true' else bool(ast.literal_eval(x)) for x in row])))
        self.data = data
        if recalc_norm:
            norm_calc = []
            for d in data:
                norm_calc.append(np.array(Image.open(os.path.join(self.dataset_path, d[0]))))
            norm_calc = np.stack(norm_calc)
            print(norm_calc.shape)
            print(torch.tensor(norm_calc, dtype=torch.float).mean([0,1,2]))
            print(torch.tensor(norm_calc, dtype=torch.float).std([0,1,2]))
        self.mean = torch.tensor([8.9478, 8.9478, 8.9478], dtype=torch.float) 
        self.std = torch.tensor([47.0021, 42.1596, 39.2562], dtype=torch.float)
        # print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx][0] + '.png'
        targets = self.data[idx][1:]
        img = Image.open(os.path.join(self.dataset_path, image))
        # img = img.resize(self.resize) #Resize image
        if self.norm:
            image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        else:
            image_to_tensor = transforms.ToTensor()
        img = image_to_tensor(img)
        target_shape = (self.resize[1],self.resize[0])#img.shape[1:]
        # print('target',target_shape)

        #Get Bounding Box Data
        
        bb_data = self.data[idx][0] + '.txt'
        x = [len(self.classes)]
        x.extend(target_shape)
        bb_target = torch.zeros(x)
        target = 0
        if bb_data in os.listdir(self.dataset_path):
            data = []
            with open(os.path.join(self.dataset_path, bb_data)) as file_obj:
                reader = csv.reader(file_obj, delimiter=' ')
                for ix, row in enumerate(reader):
                    data.append([float(x) for x in row])
            for d in data:
                if int(d[0]) == 15:  
                    target = 1
                # index = self.classes.index(d[0])
                # bb_target[index] = draw_data(target_shape, d)

        return (img, torch.tensor(targets, dtype=torch.float), torch.tensor(target, dtype=torch.float)[None])

class VisionData(torch.utils.data.DataLoader):

    def __init__(self, dataset_path, norm=False, recalc_norm=False, resize=(130,100), classes = [1,8]):
        #Norm handled in model but calculated here
        self.norm = norm
        self.dataset_path = dataset_path
        self.resize = resize
        self.classes = classes
        self.class_range = list(range(1,10))

        self.ids = [x[:-4] for x in os.listdir(self.dataset_path) if x.endswith('.png')]
        self.ids.sort()

        if recalc_norm:
            norm_calc = []
            for i in self.ids:
                norm_calc.append(np.array(Image.open(os.path.join(self.dataset_path, i + '.png'))))
            norm_calc = np.stack(norm_calc)
            print(norm_calc.shape)
            print(torch.tensor(norm_calc, dtype=torch.float).mean([0,1,2]))
            print(torch.tensor(norm_calc, dtype=torch.float).std([0,1,2]))
        self.mean = torch.tensor([0.8948, 0.8948, 0.8948], dtype=torch.float)
        self.std = torch.tensor([46.5752, 43.1358, 40.1260], dtype=torch.float)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.ids[idx] + '.png'
        targets = self.ids[idx] + '.txt'
        img = Image.open(os.path.join(self.dataset_path, image))
        tgt = np.loadtxt(os.path.join(self.dataset_path, targets), dtype=int)
        tgt_classes = tgt.copy()
        output_target = []
        for ix, c in enumerate(self.class_range):
            if c not in self.classes:
                tgt[tgt == c] = 0
            else:
                tgt[tgt == c] = ix + 1
        
        # tgt[tgt != 0] = -1
        # tgt += 1
        # output_target.append(torch.tensor(tgt, dtype=torch.float)[None])
        # for c in self.classes:
        #     tgt_temp = np.zeros(tgt_classes.shape)
        #     tgt_temp[tgt_classes == c] = 1
        #     output_target.append(torch.tensor(tgt_temp, dtype=torch.float)[None])

        # tgt = torch.cat(output_target)
        tgt = torch.tensor(tgt, dtype=torch.float)
        image_to_tensor = transforms.ToTensor()
        img = image_to_tensor(img)

        img = transforms.functional.to_pil_image(img.cpu())
        img = transforms.functional.to_tensor(transforms.Resize((100,130))(img))

        tgt = transforms.functional.to_pil_image(tgt.cpu())
        tgt = transforms.functional.to_tensor(transforms.Resize((100,130))(tgt))
        tgt.squeeze_(0)
        return (img, tgt.long())

def load_data(path_to_data, batch_size=64):
    d = AgentData(path_to_data)
    return torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True)

def load_vision_data(path_to_data, batch_size=64):
    d = VisionData(path_to_data)
    return torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True)

def load_classifier_data(path_to_data, batch_size=64):
    d = ClassifierData(path_to_data)
    return torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    data = VisionData('/Users/bsadeghian/Documents/UTCS/Deep Learning/final_deeplearning/vision_data', recalc_norm=False)
    print(data[3])
