import torch 
import torchvision
import os
import logging


def read_model(path):
    """
    Read a completely saved model(not state_dict) from path 
    """
    try:
        model = torch.load(path)
        return model
    except Exception as e:
        raise Exception(e)

def extract_layers(model, layers):
    """
    Recursively extract all layers from the model
    """
    for c in model.children():
        if sum(1 for _ in c.children()) == 0:
            layers.append(c)
        else:
            extract_layers(c, layers)
    return layers

def get_layer_properties(layer, batch_size, input_size):
    M = layer.in_channels
    N = layer.out_channels
    K = layer.kernel_size[0]
    G = layer.groups
    H = input_size[0]
    F = input_size[1]
    B = batch_size
    numerator = (B*F*H*M*N*K*K)/G
    denominator = (M*N*K*K)/G+((B*(M+N)*F*H))
    macs = B*M*N*K*K*F*H/G
    return numerator/denominator, macs

def conv_fn(layer, x, y):
    if type(layer) == torch.nn.modules.conv.Conv2d:
        M = layer.in_channels
        N = layer.out_channels
        K = layer.kernel_size[0]
        G = layer.groups
        H = y.size()[2]
        F = y.size()[3]
        B = y.size()[0]
        numerator = (B*F*H*M*N*K*K)/G
        weights = (M*N*K*K)
        memory_access_in = B*M*F*H/G
        memory_access_out = B*N*F*H/G
        memory_access = ((B*(M+N)*F*H)/G)
        denominator = weights + memory_access
        macs = B*M*N*K*K*F*H/G
        layer.ai += (numerator/denominator)
        layer.macs += macs
        layer.weights += weights
        layer.memory_access += memory_access
        layer.memory_access_in += memory_access_in
        layer.memory_access_out += memory_access_out

handler_collection = []

def add_hook(layer):
    if len(list(layer.children())) > 0:
        return

    layer.register_buffer('ai', torch.zeros(1))
    layer.register_buffer('macs', torch.zeros(1))
    layer.register_buffer('weights', torch.zeros(1))
    layer.register_buffer('memory_access', torch.zeros(1))
    layer.register_buffer('memory_access_in', torch.zeros(1))
    layer.register_buffer('memory_access_out', torch.zeros(1))
    handler = layer.register_forward_hook(conv_fn)
    handler_collection.append(handler)
    print("Registered %s" % str(layer))
