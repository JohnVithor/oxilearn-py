import torch
from safetensors.torch import load_file

net = torch.nn.Sequential(
    torch.nn.Linear(4,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,2),
    torch.nn.Softmax(dim=0)
)

target_net = torch.nn.Sequential(
    torch.nn.Linear(4,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,2),
    torch.nn.Softmax(dim=0)
)

net.load_state_dict(load_file('/home/johnvithor/oxilearn/safetensors/policy_weights.safetensors'))
target_net.load_state_dict(load_file('/home/johnvithor/oxilearn/safetensors/target_policy_weights.safetensors'))

