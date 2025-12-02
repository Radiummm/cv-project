import torch
import torch.nn as nn

# 1. 定义 AdaIN 核心功能 (这是风格迁移的灵魂)
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean

# 2. 定义轻量级编码器 (Student Encoder)
# 只有3层，比VGG小得多
class StudentEncoder(nn.Module):
    def __init__(self):
        super(StudentEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 加深加宽
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

# ======================================================
# 3. 定义【大杯】解码器 (Student Decoder - Large)
# ======================================================
class StudentDecoder(nn.Module):
    def __init__(self):
        super(StudentDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), # 对应编码器
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# 4. 组装完整的学生模型
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.encoder = StudentEncoder()
        self.decoder = StudentDecoder()

    def forward(self, content, style, alpha=1.0):
        # 提取特征
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        
        # 核心：AdaIN 风格融合
        t = adaptive_instance_normalization(content_feat, style_feat)
        
        # 控制风格强度 (alpha)
        t = alpha * t + (1 - alpha) * content_feat
        
        # 生成图片
        g_t = self.decoder(t)
        return g_t, t

# 简单测试一下模型能不能跑
if __name__ == '__main__':
    # 模拟一张图：Batch=1, Channel=3, Height=256, Width=256
    dummy_content = torch.randn(1, 3, 256, 256)
    dummy_style = torch.randn(1, 3, 256, 256)
    
    model = StudentNet()
    print("正在初始化学生模型...")
    
    # 打印参数量，看看它有多小
    total_params = sum(p.numel() for p in model.parameters())
    print(f"学生模型参数量: {total_params}")
    
    # 试运行一次
    output, _ = model(dummy_content, dummy_style)
    print(f"输出图片尺寸: {output.shape}")
    print("学生模型测试通过！")

