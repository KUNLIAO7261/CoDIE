import os
import argparse
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from siren import INF
from loss import L_exp, L_TV
from utils import get_image
from pytorch_msssim import ssim

# --------------------- 色彩空间 ---------------------
def rgb2ycbcr_torch(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299*r + 0.587*g + 0.114*b
    cb = 128 - 0.1687*r - 0.3313*g + 0.5*b  # 标准范围修正
    cr = 128 + 0.5*r - 0.4187*g - 0.0813*b
    return torch.cat([y, cb, cr], dim=1)

def ycbcr2rgb_torch(ycbcr):
    y = ycbcr[:, 0:1]
    cb = (ycbcr[:, 1:2] - 128) / 0.5  # 反向修正
    cr = (ycbcr[:, 2:3] - 128) / 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, 0, 1)  # 输出截断
def compute_dehaze_metrics(hazy_img, dehazed_img, model=None):
    # ➤ 仅用亮度通道评估
    y_hazy = 0.299*hazy_img[:,0] + 0.587*hazy_img[:,1] + 0.114*hazy_img[:,2]
    y_dehazed = 0.299*dehazed_img[:,0] + 0.587*dehazed_img[:,1] + 0.114*dehazed_img[:,2]

    # ➤ MSE → PSNR
    mse = F.mse_loss(y_dehazed, y_hazy)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    # ➤ SSIM
    ssim_val = ssim(y_dehazed.unsqueeze(1), y_hazy.unsqueeze(1), data_range=1.0).item()

    # ➤ 亮度均值
    brightness_hazy = y_hazy.mean().item()
    brightness_dehazed = y_dehazed.mean().item()

    # ➤ 参数量
    param_count = sum(p.numel() for p in model.parameters()) / 1e6 if model else 0

    return {
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim_val, 4),
        "Brightness_Hazy": round(brightness_hazy, 4),
        "Brightness_Dehazed": round(brightness_dehazed, 4),
        "Model_Params_M": round(param_count, 2)
    }

# --------------------- 工具函数 ---------------------
def get_coords(H, W):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    return torch.stack([xx, yy], dim=-1).view(-1, 2).float()

def get_patches(img, window):
    B, C, H, W = img.shape
    pad = window // 2
    img_padded = F.pad(img, [pad, pad, pad, pad], mode='reflect')
    patches = img_padded.unfold(2, window, 1).unfold(3, window, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, H*W, C*window*window)
    return patches.squeeze(0).float()

def dark_channel(x, size=15):
    pad = size // 2
    return -F.max_pool2d(-x, kernel_size=size, stride=1, padding=pad)

# --------------------- 网络结构 ---------------------
class DehazeY(INF):
    def __init__(self, patch_dim, num_layers, hidden_dim, add_layer):
        super().__init__(patch_dim, num_layers, hidden_dim, add_layer)
        # 动态大气光估计（初始值0.7，范围0.5-0.85）
        self.A = nn.Parameter(torch.tensor(0.7))  
        self.t_min = nn.Parameter(torch.tensor(0.15))  # 透射率下限

    def forward(self, patch, spatial):
        t_raw = self.output_net(torch.cat(
            [self.patch_net(patch), self.spatial_net(spatial)], dim=-1))
        t_safe = torch.sigmoid(t_raw) * 0.8 + self.t_min  # [0.15, 0.95]
        return t_safe, torch.clamp(self.A, 0.5, 0.85)
# --------------------- 主函数 ---------------------
def main():
    parser = argparse.ArgumentParser(description='Y通道图像去雾训练脚本')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--down_size', type=int, default=256)
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--L', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--delta', type=float, required=True)
    opt = parser.parse_args()

    os.makedirs(opt.output_folder, exist_ok=True)

    for fname in sorted(os.listdir(opt.input_folder)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        # ----- 加载图像与转换 -----
        img_rgb = get_image(os.path.join(opt.input_folder, fname))
        img_ycc = rgb2ycbcr_torch(img_rgb)
        y, cb, cr = img_ycc[:,0:1], img_ycc[:,1:2], img_ycc[:,2:3]

        y_lr = F.interpolate(y, size=opt.down_size, mode='bilinear', align_corners=False)
        coords = get_coords(opt.down_size, opt.down_size).cuda()
        patches = get_patches(y_lr, opt.window).cuda()

        # ----- 初始化网络 -----
        net = DehazeY(opt.window**2, 4, 256, 2).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.2)

        l_tv = L_TV(); l_exp = L_exp(16, opt.L)

        # ----- 训练循环 -----
        for epoch in range(opt.epochs):
            net.train(); optimizer.zero_grad()
            t_lr, A = net(patches, coords)
            t_lr = t_lr.view(1,1,opt.down_size,opt.down_size)
            J_lr = (y_lr - A) / (t_lr + 1e-4) + A
            J_lr = torch.clamp(J_lr, 0.05, 0.95)

            loss_spa = F.mse_loss(J_lr, y_lr)
            loss_tv  = l_tv(J_lr)
            loss_exp = l_exp(J_lr).mean()
            loss_dark= torch.mean(torch.abs(dark_channel(J_lr)))

            cb_lr = F.interpolate(cb, size=(opt.down_size, opt.down_size), mode='bilinear')
            cr_lr = F.interpolate(cr, size=(opt.down_size, opt.down_size), mode='bilinear')
            rgb_rec = ycbcr2rgb_torch(torch.cat([J_lr, cb_lr, cr_lr], dim=1))
            rgb_orig= ycbcr2rgb_torch(torch.cat([y_lr, cb_lr, cr_lr], dim=1))
            loss_chroma = F.mse_loss(rgb_rec[:,1:], rgb_orig[:,1:])
            loss_ssim = 1 - ssim(J_lr, y_lr,data_range=1.0)  # 补充到总损失
            total = (opt.alpha * loss_spa + opt.beta * loss_tv +
                     opt.gamma * loss_exp + opt.delta * loss_dark +
                     0.05 * loss_chroma + 0.2 * loss_ssim)
           

            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            if epoch % 50 == 0:
                print(f"{fname} Epoch[{epoch}/{opt.epochs}] Loss={total.item():.4f}")

        # ----- 推理阶段 -----
        net.eval(); 
        with torch.no_grad():
            coords_hr = get_coords(y.shape[2], y.shape[3]).cuda()
            patches_hr = get_patches(y, opt.window).cuda()
            start_time = time.time()
            t_hr, A = net(patches_hr, coords_hr)
            t_hr = t_hr.view(1,1,y.shape[2], y.shape[3])
            # 替换原有恢复公式
            t_hr_clamped = torch.clamp(t_hr, min=0.25,max=0.95)
            J_hr = (y - A) / (t_hr_clamped + 1e-4) + A
            y_restored = torch.clamp(J_hr, 0.05, 0.95)

            

        ycc_final = torch.cat([y_restored, cb, cr], dim=1)
        rgb_final = ycbcr2rgb_torch(ycc_final)
        infer_time = time.time() - start_time
        max_val = rgb_final.max().item()

        mean_val = rgb_final.mean().item()
        if max_val < 1e-3 or mean_val < 0.05:
            print(f" {fname} 图像亮度低: mean={mean_val:.4f}, max={max_val:.4f}")


        out_img = (torch.movedim(rgb_final[0], 0, 2).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(out_img).save(os.path.join(opt.output_folder, fname))
        metrics = compute_dehaze_metrics(img_rgb, rgb_final, model=net)
        metrics["Inference_Time_ms"] = round(infer_time * 1000, 1)

# 打印或写入日志文件
        print(f" {fname} → PSNR: {metrics['PSNR']}, SSIM: {metrics['SSIM']}, Params: {metrics['Model_Params_M']}M, Time: {metrics['Inference_Time_ms']}ms")

        print(f"已完成图像去雾: {fname}")
        print(f"透射率均值: {t_hr_clamped.mean().item():.4f}, A: {A.item():.4f}")

    print(" 所有图像处理完毕！")

if __name__ == "__main__":
    main()
