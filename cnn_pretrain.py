import torch
import torch.optim as optim
import yaml
import numpy as np

from byol.byol_pytorch import BYOL
from nn.cnn import AutoEncoder
from utils import GradualWarmupScheduler

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder(config["in_ch"], config["c_base_ch"], config["n_blk"], config["dropout"], config["piy"], config["pix"]).to(device)

# learner = BYOL(
#     model,
#     image_channel= config["in_ch"],
#     image_size = config["pix"],
#     hidden_layer = 'pool',
#     use_momentum = False       # turn off momentum in the target encoder
# ).to(device)

opt = optim.Adam(model.parameters(), lr=config["lr"])
cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=config["epochs"], eta_min=0, last_epoch=-1)
warmUpScheduler = GradualWarmupScheduler(optimizer=opt, multiplier=config["multiplier"], warm_epoch=config["epochs"] // 10, after_scheduler=cosineScheduler)

data = np.load('dataset/train.npy')
data = torch.tensor(data, dtype=torch.float32).to(device)

for epoch in range(config["epochs"]):
    loss = torch.nn.functional.mse_loss(model(data), data)
    opt.zero_grad()
    loss.backward()
    opt.step()
    warmUpScheduler.step()
    print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, LR: {opt.state_dict()['param_groups'][0]['lr']:.6f}")

# save your improved network
torch.save(model.state_dict(), 'model/cnn_model.pth')