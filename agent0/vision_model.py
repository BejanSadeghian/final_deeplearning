import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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
            
    def __init__(self, layers=[32,32,64,64,128,128], normalize=True, inference=True):
        super().__init__()

        """
        Your code here
        """
        self.normalize = normalize
        self.inference = inference
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
        # self.classifier = torch.nn.Linear(c, 3)
        

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
                    x = transforms.functional.to_tensor(transforms.Resize((100,130))(img))
                    images.append(x[None].to(device))
                x = torch.cat(images)
            else:
                img = transforms.functional.to_pil_image(x.cpu())
                x = transforms.functional.to_tensor(transforms.Resize((100,130))(img))
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
        # return self.classifier(z.mean([2,3]))

def save_model(model, name='vision'):
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

if __name__ == '__main__':
    model = Action()