import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from nn.transformer import Transformer
from nn.cnn import AutoEncoder

from utils import get_data

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    trans_model = Transformer(config["seg_len"], config["max_len"], config["t_base_ch"], config["ffn_hid"], config["n_head"], config["n_layer"], config["dropout"]).to(device)
    trans_model.load_state_dict(torch.load("model/trans_model.pth", weights_only=True, map_location=device))
    trans_model.eval()

    cnn_model = AutoEncoder(config["in_ch"], config["c_base_ch"], config["n_blk"], config["dropout"], config["piy"], config["pix"]).to(device)
    cnn_model.load_state_dict(torch.load("model/cnn_model.pth", weights_only=True, map_location=device))
    cnn_extractor = cnn_model.extractor
    cnn_extractor.eval()
    print("model load weight done.")

    dataX, dataY = get_data()

    levels = np.arange(-4.05, 4.05, 0.05)

    inputs = dataX[46].unsqueeze(0).to(device)
    yori = dataY[46].unsqueeze(0).to(device)

    # 模型预测
    B, S, C, H, W = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3), inputs.size(4)
    inputs = inputs.view(B * S, C, H, W)
    inputs = cnn_extractor(inputs)
    inputs = inputs.view(B, S, -1)

    yori = cnn_extractor(yori)
    yori = yori.view(B, 1, -1)

    outputs = trans_model.evaluate(inputs)

    print(torch.nn.functional.mse_loss(outputs, yori).item())
    yori = yori.view(B, C, 16, 8)

    inputs = inputs.view(B, S, 16, 8).cpu().numpy()
    yori = yori.view(B, C, 16, 8).cpu().numpy()
    outputs = outputs.view(B, C, 16, 8).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].contourf(inputs[0, 3], levels=levels)
    axes[0].set_title('Normalized Input')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(yori[0, 0], levels=levels)
    axes[1].set_title('Normalized Ground Truth')
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].contourf(outputs[0, 0], levels=levels)
    axes[2].set_title('Normalized Output')
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout() 
    plt.show()