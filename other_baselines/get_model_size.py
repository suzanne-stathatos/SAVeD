# N2V
import torch
import torch.nn as nn
from collections import OrderedDict
import sys

# from other_denoisers.n2v_predict import load_model

from careamics.lightning import (  
    create_careamics_module
)

def load_model(model_path):
    try: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']
        model = create_careamics_module(  
                    algorithm="n2v",
                    loss="n2v",
                    architecture="UNet",
                )
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

n2v_checkpoint_path = '/path/to/checkpoints/n2v_checkpoints/best_model.ckpt'

n2v_model = load_model(n2v_checkpoint_path)
n2v_trainable_params = count_trainable_parameters(n2v_model)
n2v_all_params = count_all_parameters(n2v_model)

print(f'N2V trainable parameters: {n2v_trainable_params}, all parameters: {n2v_all_params}')
# print(f'Ratio: {best_trainable_params / n2v_trainable_params}')
