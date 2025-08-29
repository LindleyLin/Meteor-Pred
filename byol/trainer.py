from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from .byol_pytorch import BYOL


def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# class

# main trainer

class BYOLTrainer(Module):
    def __init__(
        self,
        net: Module,
        *,
        image_size: int,
        hidden_layer: str,
        learning_rate: float,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int = 16,
        optimizer_klass = Adam,
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        byol_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
    ):
        super().__init__()

        self.rank = 0
        self.is_main_process = True

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型和数据移动到设备上
        self.byol = BYOL(net, image_size=image_size, hidden_layer=hidden_layer, **byol_kwargs).to(self.device)
        self.optimizer = optimizer_klass(self.byol.parameters(), lr=learning_rate, **optimizer_kwargs)

        # DataLoader
        self.dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        self.num_train_steps = num_train_steps
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)

        if self.is_main_process:
            self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
            assert self.checkpoint_folder.is_dir()
        
        # 注册一个buffer来保存步数
        self.register_buffer('step', torch.tensor(0))

    def print(self, msg):
        if self.is_main_process:
            print(msg)

    def forward(self):
        step = self.step.item()
        data_it = cycle(self.dataloader)
        
        for _ in range(self.num_train_steps):
            images = next(data_it)

            # 将数据移动到设备上
            images = images.to(self.device)

            # 训练逻辑
            # 使用 autocast 手动实现混合精度
            with torch.amp.autocast():
                loss = self.byol(images)
            
            # 反向传播
            loss.backward()

            self.print(f'loss {loss.item():.3f}')

            # 优化器更新
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 更新 moving average
            self.byol.update_moving_average()

            # 保存模型
            if not (step % self.checkpoint_every) and self.is_main_process:
                checkpoint_num = step // self.checkpoint_every
                checkpoint_path = self.checkpoint_folder / f'checkpoint.{checkpoint_num}.pt'
                model_to_save = self.byol.net
                torch.save(model_to_save.state_dict(), str(checkpoint_path))

            step += 1

        self.print('training complete')