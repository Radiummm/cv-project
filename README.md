
# Lightweight Style Transfer Project (轻量级任意风格迁移)

本项目实现了一个基于**知识蒸馏**的超轻量级风格迁移模型。我们将笨重的 AdaIN 教师模型压缩了 **50+ 倍**，实现了 **30FPS+** 的实时推理速度。

## 🛠️ 1. 环境部署 (Environment Setup)

### 第一步：克隆代码

```bash
git clone https://github.com/Radiummm/cv-project.git
cd cv-project
```

### 第二步：安装依赖

确保你的环境中有 Python 3.8+ 和 PyTorch。

```bash
pip install -r requirements.txt
```

-----

## 💾 2. 数据准备 (Data Preparation)

**注意：** 数据集文件较大，请手动下载并严格按照以下目录结构放置。

### 下载链接

1.  **内容图 (Content Images):**
      * 使用 **COCO 2017 Validation Set** (约 1GB)。
      * [点击下载 COCO Val2017](https://www.google.com/search?q=http://images.cocodataset.org/zips/val2017.zip)
2.  **风格图 (Style Images):**
      * 使用 **WikiArt** 或 **Kaggle Best Artworks** (约 2GB)。
      * [点击下载 Kaggle Artworks](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)

### 目录结构 (必须一致！)

请在项目根目录下创建 `data` 文件夹，解压后如下所示：

```text
/cv-project
├── data/
│   ├── content/       <-- 把解压后的 COCO 图片(.jpg)全部放在根目录下
│   │   ├── 000000000139.jpg
│   │   └── ...
│   └── style/         <-- 把解压后的 风格 图片(.jpg)全部放在根目录下
│       ├── monet.jpg
│       └── ...
├── student_model/     <-- 我们的核心代码
│   ├── checkpoints/   <-- 存放训练好的模型权重 (student_latest.pth)
│   ├── net.py         <-- 网络结构
│   ├── train.py       <-- 训练脚本
│   └── test_student.py <-- 推理脚本
└── ...
```

-----

## 🚀 3. 如何运行 (Usage)

### 🎨 生成风格化图片 (推理测试)

使用训练好的轻量级模型进行风格迁移：

```bash
cd student_model

# 基础命令格式
python test_student.py --content <内容图路径> --style <风格图路径> --output <保存路径>

# 示例：生成一张莫奈风格的图
python test_student.py \
  --content ../data/content/000000000139.jpg \
  --style ../data/style/Claude_Monet_1.jpg \
  --output result.jpg
```

### 🏋️‍♂️ 重新训练 (可选)

如果你想复现蒸馏过程：

```bash
cd student_model
python train.py
```



## 📊 性能数据 (Benchmark)

  * **模型大小:** 10.5 MB (相比教师模型压缩 50 倍)
  * **推理速度:** \~30 ms/img (On Tesla T4)
  * **FPS:** 33+ (实现实时视频流处理)
