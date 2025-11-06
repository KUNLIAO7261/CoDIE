import os, csv, time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
from pytorch_msssim import ssim

class Evaluator:
    def __init__(self, input_folder, output_folder, model=None, device='cuda', output_csv='dehaze_metrics.csv'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device) if model else None
        self.output_csv = output_csv
        self.headers = ['filename', 'PSNR', 'SSIM', 'Brightness_orig', 'Brightness_dehazed', 'Model_Params_M', 'Inference_Time_ms']

    def count_parameters(self):
        if self.model:
            return round(sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6, 2)
        return 0.0

    def compute_psnr(self, mse):
        return round(20 * torch.log10(1.0 / torch.sqrt(mse)).item(), 2)

    def process_image(self, fname):
        input_path = os.path.join(self.input_folder, fname)
        output_path = os.path.join(self.output_folder, fname)

        if not os.path.exists(output_path):
            print(f"❌ 去雾结果缺失: {fname}")
            return None

        # 加载并转为张量
        img_orig = to_tensor(Image.open(input_path).convert('RGB')).unsqueeze(0).to(self.device)
        img_rest = to_tensor(Image.open(output_path).convert('RGB')).unsqueeze(0).to(self.device)

        # 提取亮度通道
        y_orig = 0.299*img_orig[:,0] + 0.587*img_orig[:,1] + 0.114*img_orig[:,2]
        y_rest = 0.299*img_rest[:,0] + 0.587*img_rest[:,1] + 0.114*img_rest[:,2]

        # 计算指标
        mse = F.mse_loss(y_rest, y_orig)
        psnr = self.compute_psnr(mse)
        ssim_val = round(ssim(y_rest.unsqueeze(1), y_orig.unsqueeze(1), data_range=1.0).item(), 4)
        brightness_orig = round(y_orig.mean().item(), 4)
        brightness_rest = round(y_rest.mean().item(), 4)

        # 计算推理时间
        start = time.time()
        if self.model:
            with torch.no_grad():
                _ = self.model(img_orig)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        infer_time = round((time.time() - start) * 1000, 1)  # ms

        params = self.count_parameters()

        return [fname, psnr, ssim_val, brightness_orig, brightness_rest, params, infer_time]

    def evaluate_all(self):
        files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

            for fname in sorted(files):
                result = self.process_image(fname)
                if result:
                    writer.writerow(result)
                    print(f"✅ 已评估: {fname}")

        print(f"\n📁 所有图像评估完成，结果保存在: {self.output_csv}")
