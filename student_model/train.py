import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, PngImagePlugin
from pathlib import Path
from tqdm import tqdm
import warnings
import os

# 导入你之前上传的 student_model/net.py
from net import StudentNet, calc_mean_std

# 忽略警告
warnings.filterwarnings("ignore")
# 增大 PIL 加载大图的限制
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
Image.MAX_IMAGE_PIXELS = None

# ==========================================
# 1. 配置区域 (Config)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
BATCH_SIZE = 8       # 如果显存不够报错，改小成 4
MAX_ITER = 50000     # 训练总步数
SAVE_INTERVAL = 1000 # 每1000步保存一次
LOG_INTERVAL = 10    # 每10步打印一次

# 关键路径 (根据你的目录结构)
VGG_PATH = "../teacher_model/models/vgg_normalised.pth"
CONTENT_DIR = "../data/content"
STYLE_DIR = "../data/style"
SAVE_DIR = "checkpoints"

# ==========================================
# 2. 定义 VGG 裁判 (用于计算 Loss)
# ==========================================
class VGGEncoder(nn.Module):
    def __init__(self, model_path):
        super(VGGEncoder, self).__init__()
        # 标准 AdaIN 的 VGG 结构
        vgg_layers = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, 3), nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3), nn.ReLU(), # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, 3), nn.ReLU(), # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3), nn.ReLU(), # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, 3), nn.ReLU(), # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3), nn.ReLU(), # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3), nn.ReLU(), # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3), nn.ReLU(), # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, 3), nn.ReLU(), # relu4-1
        )
        # 加载权重
        vgg_layers.load_state_dict(torch.load(model_path),strict=False)
        
        # 切分网络
        self.enc_1 = vgg_layers[:4]
        self.enc_2 = vgg_layers[4:11]
        self.enc_3 = vgg_layers[11:18]
        self.enc_4 = vgg_layers[18:31]

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = []
        x = self.enc_1(x)
        results.append(x)
        x = self.enc_2(x)
        results.append(x)
        x = self.enc_3(x)
        results.append(x)
        x = self.enc_4(x)
        results.append(x)
        return results

# ==========================================
# 3. 数据集加载器
# ==========================================
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        # 兼容 jpg, png, jpeg
        self.paths = list(Path(self.root).glob('*.[jJpP][pPnN][gG]*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            img = Image.open(str(path)).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            # 如果读到坏图，返回一张随机图防止崩溃
            print(f"坏图跳过: {path}")
            return torch.randn(3, 512, 512)

    def __len__(self):
        return len(self.paths)

def infinite_iter(loader):
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)
            yield next(it)

# ==========================================
# 4. 损失函数 (蒸馏核心)
# ==========================================
def calc_content_loss(input, target):
    return nn.MSELoss()(input, target)

def calc_style_loss(input, target):
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return nn.MSELoss()(input_mean, target_mean) + \
           nn.MSELoss()(input_std, target_std)

# ==========================================
# 5. 主程序
# ==========================================
def train():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    print(f"正在加载数据... Content: {CONTENT_DIR}, Style: {STYLE_DIR}")
    content_set = FlatFolderDataset(CONTENT_DIR, transform)
    style_set = FlatFolderDataset(STYLE_DIR, transform)
    
    content_loader = DataLoader(content_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    style_loader = DataLoader(style_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    content_iter = infinite_iter(content_loader)
    style_iter = infinite_iter(style_loader)
    print(f"加载完成! 内容图: {len(content_set)}, 风格图: {len(style_set)}")

    # 模型初始化
    print("正在初始化模型...")
    vgg = VGGEncoder(VGG_PATH).to(DEVICE)
    student = StudentNet().to(DEVICE)
    student.train()
    
    optimizer = optim.Adam(student.parameters(), lr=LR)

    print("=== 开始蒸馏训练 ===")
    pbar = tqdm(range(MAX_ITER))
    for i in pbar:
        # 1. 获取数据
        content_images = next(content_iter).to(DEVICE)
        style_images = next(style_iter).to(DEVICE)

        # 2. 学生作画
        g_t, t = student(content_images, style_images)

        # 3. 老师打分 (Loss)
        g_t_feats = vgg(g_t)
        style_feats = vgg(style_images)
        content_feat = vgg(content_images)[-1]

        loss_c = calc_content_loss(g_t_feats[-1], content_feat)
        loss_s = 0
        for g, s in zip(g_t_feats, style_feats):
            loss_s += calc_style_loss(g, s)
        
        loss_total = loss_c + 10.0 * loss_s

        # 4. 优化
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 5. 记录
        if (i + 1) % LOG_INTERVAL == 0:
            pbar.set_description(f"Loss C: {loss_c.item():.3f} | Loss S: {loss_s.item():.3f}")

        if (i + 1) % SAVE_INTERVAL == 0:
            torch.save(student.state_dict(), f"{SAVE_DIR}/student_latest.pth")

if __name__ == "__main__":
    train()
