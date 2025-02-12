import numpy as np
import torch
import pickle

class NNConfig:
    def __init__(self, config):
        self.in_features = int(config["in_features"])
        self.out_features = int(config["out_features"]) 
        self.gamma = config["Lipschitz_constant"]
        self.layer = config["layer"] 
        self.num_layers = config["num_layers"]
        self.activations = [config["activations"]] * (self.num_layers-1)
        self.widths = [self.in_features] + [config["width_each_layer"]] * (self.num_layers-1)\
                     + [self.out_features]
        
def seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_nn_config(config):
    return NNConfig(config)

def save_nn_weights(nn, full_path):
    torch.save(nn.state_dict(), full_path)

def load_nn_weights(nn, full_path, device):
    loaded_state_dict = torch.load(full_path, map_location=device, weights_only=True)
    nn.load_state_dict(loaded_state_dict)
    nn.to(device)
    return nn

def save_dict(dict_obj, fullname):
    with open(fullname, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fullname):
    with open(fullname, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    return loaded_obj

def dict2func(d):
    return lambda x: d[x]

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_grad_l2_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm