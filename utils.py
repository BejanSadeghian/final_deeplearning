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
        if bb_data in os.listdir(self.dataset_path):
            data = []
            with open(os.path.join(self.dataset_path, bb_data)) as file_obj:
                reader = csv.reader(file_obj, delimiter=' ')
                for ix, row in enumerate(reader):
                    data.append([float(x) for x in row])
            for d in data:
                index = self.classes.index(d[0])
                bb_target[index] = draw_data(target_shape, d)

        return (img, torch.tensor(targets, dtype=torch.float), bb_target)

def draw_data(mat_shape, yolo, min_val=5):
    mat = np.zeros(mat_shape)
    mat_y, mat_x = mat_shape
    x = int(yolo[1] * mat_x)
    y = int(yolo[2] * mat_y)
    w = int(yolo[3] * mat_x) if int(yolo[3] * mat_x) > min_val else min_val 
    h = int(yolo[4] * mat_y) if int(yolo[4] * mat_y) > min_val else min_val
    w = w if w%2 == 0 else w + 1
    h = h if h%2 == 0 else h + 1
    
    # overlay = np.ones((h,w))
    # overlay_crop = mat[int(max(y-(h/2), 0)) : int(min(y+(h/2), mat_y)), int(max(x-(w/2), 0)) : int(min(x+(w/2), mat_x))].shape
    # overlay = overlay[:overlay_crop[0], :overlay_crop[1]]
    mat[int(max(y-(h/2), 0)) : int(min(y+(h/2), mat_y)), int(max(x-(w/2), 0)) : int(min(x+(w/2), mat_x))] = 1.0
    return torch.tensor(mat, dtype=torch.float)

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
    data = VisionData('/Users/bsadeghian/Documents/UTCS/Deep Learning/deeplearning/final/train_data')
    print(data[473])
    # import matplotlib.pyplot as plt
    # print(data[0][2])
    # plt.imshow(data[0][2].numpy())
    # mat = np.zeros((10,10))
    # yolo = [1, 0.2, 0.2, 5, 5]
    # print(mat)
    # print(draw_data(mat, yolo))