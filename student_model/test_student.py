import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import argparse
from net import StudentNet # 导入你的模型结构

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--output', type=str, default='result_student.jpg')
    parser.add_argument('--model', type=str, default='checkpoints/student_latest.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 预处理 (固定大小以便测速)
    trans = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    # 加载图片
    try:
        content = trans(Image.open(args.content).convert('RGB')).unsqueeze(0).to(device)
        style = trans(Image.open(args.style).convert('RGB')).unsqueeze(0).to(device)
    except Exception as e:
        print(f"图片读取失败: {e}")
        return

    # 2. 加载模型
    print(f"Loading model: {args.model} (Size: 0.7MB!)")
    student = StudentNet().to(device)
    student.load_state_dict(torch.load(args.model, map_location=device))
    student.eval()

    # 3. 测速 (预热)
    with torch.no_grad():
        _ = student(content, style) 
        
        # 正式计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        output, _ = student(content, style)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

    print(f"=== 推理耗时: {(end - start)*1000:.2f} ms ===")

    # 4. 保存
    from torchvision.utils import save_image
    save_image(output, args.output)
    print(f"结果已保存: {args.output}")

if __name__ == '__main__':
    test()
