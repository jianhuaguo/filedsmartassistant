# testvalAI

#### 介绍
{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

#### 软件架构
软件架构说明
pest_detection_yolo/
├── data/                      # 数据集目录
│   ├── images/                # 图片目录
│   │   ├── IP000000000.jpg
│   │   └── ...
│   └── annotations/           # XML标注文件目录
│       ├── IP000000000.xml
│       └── ...
├── datasets/                  # 转换后的YOLO格式数据集
│   ├── train/                 # 训练集
│   │   ├── images/
│   │   └── labels/
│   ├── val/                   # 验证集
│   │   ├── images/
│   │   └── labels/
│   └── test/                  # 测试集
│       ├── images/
│       └── labels/
├── models/                    # 模型配置文件
│   └── pest_yolov5.yaml       # 模型结构配置
├── runs/                      # 训练结果（自动生成）
├── weights/                   # 模型权重文件
├── src/                       # 源代码
│   ├── convert_xml_to_yolo.py # XML转YOLO格式
│   ├── split_dataset.py       # 划分训练/验证/测试集
│   ├── train.py               # 训练脚本
│   ├── detect.py              # 推理脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数
├── app/                       # Web应用
│   ├── app.py
│   ├── static/
│   └── templates/
├── classes.txt                # 类别名称（0-101对应中文）
├── dataset.yaml               # 数据集路径配置
└── requirements.txt      

#### 安装教程

pip install -r requirements.txt

#### 使用说明

环境准备
pip install -r requirements.txt

conda create -n pest_recognition_env python=3.10.18

conda activate pest_recognition_env

数据预处理
bash
# 1. 将XML标注转换为YOLO格式
python -m src.convert_xml_to_yolo

# 2. 划分训练/验证/测试集
python -m src.split_dataset



模型训练
bash
python -m src.train --epochs 100 --batch 32


训练完成后，最佳模型会保存在 runs/train/pest_detection/weights/best.pt，建议复制到 weights/ 目录下
模型评估
bash
python -m src.evaluate --model weights/best.pt


单图检测
bash
python -m src.detect --model weights/best.pt --img path/to/image.jpg

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
