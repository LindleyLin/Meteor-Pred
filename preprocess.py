import numpy as np
import yaml
from scipy.ndimage import zoom

data = np.load("dataset/original.npy")

b, h, w = data.shape
target_h, target_w = 32, 32

processed_images = []
for i in range(b):
    zoom_factor_h = target_h / h
    zoom_factor_w = target_w / w

    resized_array = zoom(data[i, :, :], (zoom_factor_h, zoom_factor_w), order=1)
    processed_images.append(resized_array)

processed_data = np.stack(processed_images, axis=0)

mean = np.mean(processed_data)
std = np.std(processed_data)

if std > 1e-6:
    normalized_data = (processed_data - mean) / std
else:
    normalized_data = processed_data - mean

final_data = np.expand_dims(normalized_data, axis=1)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

k = final_data.shape[0]
sequence_length = config["seq_len"]

train_x = []
train_y = []

for i in range(k - sequence_length):
    x_sequence = final_data[i : i + sequence_length]
    train_x.append(x_sequence)
    
    y_target = final_data[i + sequence_length]
    train_y.append(y_target)

train_x = np.array(train_x)
train_y = np.array(train_y)

np.save('dataset/train_x.npy', train_x)
np.save('dataset/train_y.npy', train_y)