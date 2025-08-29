import torch
import torch.optim as optim
import yaml

from nn.diffusion import GaussianDiffusionTrainer
from nn.cnn import AutoEncoder
from nn.transformer import Transformer
from nn.ddpm import UNet
from utils import GradualWarmupScheduler, get_loader

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_loader, _ = get_loader()

# model setup
trans_model = Transformer(config["seg_len"], config["max_len"], config["t_base_ch"], config["ffn_hid"], config["n_head"], config["n_layer"], config["dropout"]).to(device)
trans_model.load_state_dict(torch.load("model/trans_model.pth", weights_only=True, map_location=device))
trans_model.eval()

cnn_model = AutoEncoder(config["in_ch"], config["c_base_ch"], config["n_blk"], config["dropout"], config["piy"], config["pix"]).to(device)
cnn_model.load_state_dict(torch.load("model/cnn_model.pth", weights_only=True, map_location=device))
cnn_extractor = cnn_model.extractor
cnn_extractor.eval()

net_model = UNet(config["in_ch"], config["d_base_ch"], config["ch_mult"], config["attn"], config["n_res"], config["cdim"], config["dropout"]).to(device)
    
# net_model.load_state_dict(torch.load("model/model.pth", weights_only=True, map_location=device))

optimizer = torch.optim.AdamW(net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"], eta_min=0, last_epoch=-1)
warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["multiplier"], warm_epoch=config["epochs"] // 10, after_scheduler=cosineScheduler)
trainer = GaussianDiffusionTrainer(net_model, config["T"]).to(device)
train_loss = 0

seg_len = config["seg_len"]

# start training
for e in range(config["epochs"]):
    for batch_idx, (data, target) in enumerate(train_loader):
        # train
        optimizer.zero_grad()
        c, x_0 = data.to(device), target.to(device)
        with torch.no_grad():
            B, S, C, H, W = c.size(0), c.size(1), c.size(2), c.size(3), c.size(4)
            c = c.view(B * S, C, H, W)
            c = cnn_extractor(c)
            c = c.view(B, S, -1)

            c = trans_model.evaluate(c)

        loss = trainer(x_0, c)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net_model.parameters(), config["grad_clip"])
        optimizer.step()
    warmUpScheduler.step()
    torch.save(net_model.state_dict(), "model/ddpm_model.pth")
    print(f"Epoch: {e}, Loss: {train_loss / len(train_loader):.4f}, LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
    train_loss = 0