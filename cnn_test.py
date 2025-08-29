import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from nn.cnn import AutoEncoder

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    model = AutoEncoder(config["in_ch"], config["c_base_ch"], config["n_blk"], config["dropout"], config["piy"], config["pix"]).to(device)
    model.load_state_dict(torch.load("model/cnn_model.pth", weights_only=True, map_location=device))
    print("model load weight done.")
    model.eval()

    data = np.load('dataset/train.npy')

    levels = np.arange(-4.05, 4.05, 0.05)

    inputs = torch.tensor(data[49], dtype=torch.float).unsqueeze(0).to(device)

    sampledImgs = model(inputs)
    print(torch.nn.functional.mse_loss(sampledImgs, inputs).item())

    inputs = inputs.cpu().numpy()
    outputs = sampledImgs.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    im1 = axes[0].contourf(inputs[0, 0], levels=levels)
    axes[0].set_title('Normalized Input')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(outputs[0, 0], levels=levels)
    axes[1].set_title('Normalized Output')
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()