import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):

    def __init__(self, d_model, dim):
        super().__init__()
        self.register_buffer(
            'inv_freq', 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        if not t.shape:
            t = t.unsqueeze(0)
            
        t = t.float().unsqueeze(1)
    
        sin_emb = torch.sin(t * self.inv_freq)
        cos_emb = torch.cos(t * self.inv_freq)
        
        emb = torch.cat([sin_emb, cos_emb], dim=1)
    
        emb = self.mlp(emb)
        return emb


class DownSample(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x, temb, cemb):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, temb, cemb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_ch, c_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        
        self.proj_q_self = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k_self = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v_self = nn.Conv2d(in_ch, in_ch, 1)

        self.proj_q_cross = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k_cross = nn.Linear(c_dim, in_ch)
        self.proj_v_cross = nn.Linear(c_dim, in_ch)

        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x, cemb):
        B, C, H, W = x.shape
        h = self.group_norm(x)

        q_self = self.proj_q_self(h)
        k_self = self.proj_k_self(h)
        v_self = self.proj_v_self(h)

        q_self = q_self.permute(0, 2, 3, 1).view(B, H * W, C)
        k_self = k_self.view(B, C, H * W)
        w_self = torch.bmm(q_self, k_self) * (C ** (-0.5))
        w_self = F.softmax(w_self, dim=-1)

        v_self = v_self.permute(0, 2, 3, 1).view(B, H * W, C)
        h_self = torch.bmm(w_self, v_self)

        h_self = h_self.view(B, H, W, C).permute(0, 3, 1, 2)
        h = h + h_self

        q_cross = self.proj_q_cross(h) 
        k_cross = self.proj_k_cross(cemb)
        v_cross = self.proj_v_cross(cemb)

        q_cross = q_cross.permute(0, 2, 3, 1).view(B, H * W, C)
        w_cross = torch.bmm(q_cross, k_cross.permute(0, 2, 1)) * (C ** (-0.5))
        w_cross = F.softmax(w_cross, dim=-1)

        h_cross = torch.bmm(w_cross, v_cross)

        h_cross = h_cross.view(B, H, W, C).permute(0, 3, 1, 2)
        h_final = self.proj(h_cross)

        return x + h_final 


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, cdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn_block = AttnBlock(out_ch, cdim)
        else:
            self.attn_block = nn.Identity()
        self.attn = attn

    def forward(self, x, temb, cemb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        if self.attn:
            h = self.attn_block(h, cemb)
        return h


class UNet(nn.Module):
    def __init__(self, ori_ch, ch, ch_mult, attn, num_res_blocks, cdim, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(ch, tdim)

        self.head = nn.Conv2d(ori_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim, cdim=cdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, cdim, dropout, True),
            ResBlock(now_ch, now_ch, tdim, cdim, dropout, False)
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, cdim=cdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, ori_ch, 3, stride=1, padding=1)
        )

    def forward(self, x, t, cemb):
        temb = self.time_embedding(t)

        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb, cemb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    model = UNet(ori_ch=1, ch=64, ch_mult=[1, 2, 2], attn=[0, 1], num_res_blocks=2, cdim=128, dropout=0.1)
    x = torch.randn(2, 1, 64, 64)
    t = torch.randint(500, (2, ))
    cemb = torch.randn(2, 2, 128)
    y = model(x, t, cemb)
    torch.save(model.state_dict(), "model/test_model.pth")
    print(y.shape)