import torch
from hrnet import HighResolutionNet

# def get_model(model_path):   # Original code.

#     model=HighResolutionNet()

#     loaded = torch.load(model_path)

#     if isinstance(loaded, torch.nn.Module):  # if it's a full model already
#         model.load_state_dict(loaded.state_dict())
#     else:
#         model.load_state_dict(loaded)
#     return model




### Code help from Chat-GPT

def get_model(model_path):
    model = HighResolutionNet()

    # Map the model to CPU if CUDA is not available
    if not torch.cuda.is_available():
        loaded = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        loaded = torch.load(model_path)

    # Load the state dict properly depending on the type of 'loaded'
    if isinstance(loaded, torch.nn.Module):  # if it's a full model already
        model.load_state_dict(loaded.state_dict())
    else:
        model.load_state_dict(loaded)
    
    return model
