# FADNet: Frequency-Aware Diffusion Network for OCTA Image Denoising

> **Iris** — Diffusion network method for OCTA image denoising

## 项目结构

```
Iris/
├── v1.0/                          # 原始版本（GitHub 初始提交）
│   ├── model.py                   # FADNet 模型架构
│   ├── diffusion.py               # 扩散过程
│   ├── unet.py                    # U-Net 骨干网络
│   ├── cross_attention.py         # 交叉注意力模块
│   ├── mhfb.py                    # 多头频率分支模块
│   ├── sfe.py                     # 浅层特征提取
│   ├── pretrain_sfe.py            # SFE 预训练
│   ├── dwt_module.py              # 离散小波变换模块
│   ├── tawg.py                    # Taylor 系列小波引导
│   ├── dataset.py                 # 数据集加载
│   ├── train.py                   # 训练脚本
│   ├── inference.py               # 推理脚本
│   ├── draw_architecture.py       # 架构图绘制
│   ├── extract_pdf.py             # PDF 文本提取
│   └── image/                     # 结果图像
│
└── v1.1/                          # 更新版本（本地开发版）
    ├── evaluation/                # 评估模块（新增）
    │   ├── __init__.py
    │   └── metrics.py
    ├── test_subset/               # 测试数据子集（新增）
    └── ...                        # 以上所有 v1.0 文件 + 更新
```

## 版本说明

| 版本 | 说明 |
|------|------|
| **v1.0** | 初始版本，包含 FADNet 核心架构、扩散模型、训练/推理流程 |
| **v1.1** | 开发版本，新增评估模块（evaluation/）、测试子集（test_subset/），模型和训练代码有所更新 |

## 环境依赖

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- opencv-python
- tqdm
- tensorboard

## 快速开始

```bash
# 训练
cd v1.1
python train.py

# 推理
python inference.py --checkpoint <path_to_checkpoint>
```

## 许可证

This project is for research purposes.
