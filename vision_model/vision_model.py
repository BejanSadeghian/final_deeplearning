import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class Vision(torch.nn.Module):
    
    class conv_block(torch.nn.Module):
        def __init__(self, channel_in, channel_out, stride=2, kernel_size=3, dilation=2):
            super().__init__()
            self.c1 = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=kernel_size//2)
            self.b1 = nn.BatchNorm2d(channel_out)
            self.c2 = nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size, dilation=dilation, stride=1, padding=kernel_size//2)
            self.b2 = nn.BatchNorm2d(channel_out)
            
            self.downsample = None
            if channel_in != channel_out or stride != 1:
                self.downsample = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride, dilation=dilation)
            
        def forward(self, x):
            self.activation = F.relu(self.b2(self.c2(self.b1(self.c1(x))))) #consider adding relus between
            identity = x
            if self.downsample != None:
                identity = self.downsample(identity)
            return self.activation + identity
            
    class upconv_block(torch.nn.Module):
        
        def __init__(self, channel_in, channel_out, stride=2, kernel_size=3, dilation=2):    
            super().__init__()
            self.upsample = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, output_padding=1)
        
        def forward(self, x, output_pad=False):
            return F.relu(self.upsample(x))
            
    def __init__(self, layers=[32,32,64,64,128,128], normalize=True, inference=True, classes=[1,8]):
        super().__init__()

        """
        Your code here
        """
        self.CLASSES = classes
        self.normalize = normalize
        self.inference = inference
        self.resize = (100,130)
        self.puck_layer = classes.index(8) + 1 #Other group is 0 so we add +1
        print('Puck Layer found at {}'.format(self.puck_layer))
        print('Forward Pass Auto Resize Image: {}, {}'.format(self.normalize, self.resize))
        print('Forward Pass Auto Normalize Image: {}'.format(self.normalize))

        self.mean = torch.tensor([8.9478, 8.9478, 8.9478], dtype=torch.float) 
        self.std = torch.tensor([47.0021, 42.1596, 39.2562], dtype=torch.float)

        c = 3        
        self.network = torch.nn.ModuleList()
        for l in layers:
            kernel_size = 7 if c == 3 else 3
            stride = 1 if c == 3 else 2
            self.network.append(self.conv_block(c, l, stride, kernel_size, 1))
            c = l
        
        self.upnetwork = torch.nn.ModuleList()
        self.upnetwork.append(self.upconv_block(c, layers[-2]))
        c = layers[-2]
        for l in reversed(layers[:-2]):
            self.upnetwork.append(self.upconv_block(c * 2, l, 2, 3, 1)) # x2 input because of skip
            c = l
        self.classifier = torch.nn.Conv2d(c, 3, kernel_size=1)        

    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        ##Add preprocessing
        if self.inference:
            # print('inference')
            device = x.device
            x = x.squeeze()
            if len(x.shape) == 4:
                images = []
                for i in x:
                    img = transforms.functional.to_pil_image(i.cpu())
                    x = transforms.functional.to_tensor(transforms.Resize(self.resize)(img))
                    images.append(x[None].to(device))
                x = torch.cat(images)
            else:
                img = transforms.functional.to_pil_image(x.cpu())
                x = transforms.functional.to_tensor(transforms.Resize(self.resize)(img))
                x = x[None].to(device)

        if self.normalize:
            x = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device) 
        
        activations = []
        for i, layer in enumerate(self.network):
            z = layer(x)
            activations.append(z)
            x = z
        z = self.upnetwork[0](x)
        for i, layer in enumerate(self.upnetwork[1:]):
            x = torch.cat([z[:,:, :activations[-2-i].size(2), :activations[-2-i].size(3)], activations[-2-i]], dim=1)
            z = layer(x)
        return self.classifier(z)

    def to_multi_channel(self, heatmap, class_range=list(range(1,10)), classes = None):
        """
        heatmap shape (C,H,W)
        """    
        def to_single_channel(heatmap):
            heatmap = heatmap.float()
            return ((torch.exp(heatmap.cpu()) / torch.exp(heatmap.cpu()).sum(0)).max(0).indices)

        ## For BCE Loss
        heatmap = heatmap.float()
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
            classes = list(range(len(self.CLASSES)+1))

        for c in classes:
            tgt_temp = np.zeros(tgt_classes.shape)
            tgt_temp[tgt_classes == float(c)] = 1
            output_target.append(torch.tensor(tgt_temp, dtype=torch.float)[None])

        tgt = torch.cat(output_target)
        return tgt

    def extract_peak(self, heatmap, max_pool_ks=7, min_score=0.4, max_det=100):
        """
        Your code here.
        Extract local maxima (peaks) in a 2d heatmap.
        @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
        @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
        @min_score: Only return peaks greater than min_score
        @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                    heatmap value at the peak. Return no more than max_det peaks per image
        """
        min_avg = min_score / max_pool_ks #normalize to size of the kernel
        H,W = heatmap.size()

        avg_map = F.avg_pool2d(heatmap[None,None], kernel_size=max_pool_ks, stride=1, padding=max_pool_ks//2)
        avg_map.squeeze_()
        if avg_map.max().item() > min_avg:
            arg_max_map = avg_map.argmax()
            y = arg_max_map // avg_map.shape[1]
            x = arg_max_map % avg_map.shape[1]
            return(True,y.item(),x.item())
        else:
            return(False,None,None)

    def detect(self, heatmap, sigmoid=True):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        heatmap.squeeze_(0)
        if sigmoid:
            heatmap = self.to_multi_channel(heatmap)
        max_vals = heatmap[self.puck_layer,:,:]      
        ultimate_res = self.extract_peak(max_vals, max_pool_ks=15) 
        
        return ultimate_res

def save_vision_model(model, name='vision'):
    from torch import save
    from os import path
    if isinstance(model, Vision):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '{}.th'.format(name)))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_vision_model(name='vision'):
    from torch import load
    from os import path
    r = Vision()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '{}.th'.format(name)), map_location='cpu'))
    return r

    