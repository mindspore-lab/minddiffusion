## 目录

[Check English](./README_EN.md)

- [Wukong-Huahua](#wukong-huahua模型)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
  - [准备checkpoint](#准备checkpoint)
  - [文图生成](#文图生成)
  - [生成样例](#生成样例)

## Wukong-Huahua 悟空画画模型

Wukong-Huahua是基于扩散模型的中文文生图大模型，由**华为诺亚团队**携手**中软分布式并行实验室**，**昇腾计算产品部**联合开发。它基于[Wukong dataset](https://wukong-dataset.github.io/wukong-dataset/)训练得到，并使用昇思框架实现。

## 环境要求

- 硬件
  - 准备Ascend处理器搭建硬件环境
- 框架
  - [Mindspore](https://www.mindspore.cn/ "Mindspore") >= 1.9
  - 其他Python包需求请参考[requirements.txt](./requirements.txt)
- 如需查看详情，请参考如下资源
  - [Mindspore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [Mindspore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速开始

### 准备checkpoint

下载wukong-huahua预训练参数 [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) 至 wukong-huahua/models/ 目录.

### 文图生成

要进行文图生成，可以运行txt2img.py 或者直接使用默认参数运行 infer.sh.

```shell
python txt2img.py --prompt [input text] --ckpt_path [ckpt_path] --H [image_height] --W [image_width] --outdir [image save folder] --n_samples [number of images to generate] --plms --skip_grid
```

```shell
bash infer.sh
```

更高的分辨率需要更大的显存. 对于 Ascend 910 卡, 我们可以同时生成2张1024x768的图片或者16张512x512的图片。

### 生成样例

下面是我们的wukong-huahua模型生成的一些样例以及对应的`[input text]`。

```
城市夜景 赛博朋克 格雷格·鲁特科夫斯基
```

![城市夜景 赛博朋克 格雷格·鲁特科夫斯基](demo/城市夜景%20赛博朋克%20格雷格·鲁特科夫斯基.png)

```
莫奈 撑阳伞的女人 月亮 梦幻
```

![莫奈 撑阳伞的女人 月亮 梦幻](demo/莫奈%20撑阳伞的女人%20月亮%20梦幻.png)

```
海上日出时候的奔跑者
```

![海上日出时候的奔跑者](demo/海上日出时候的奔跑者.png)

```
诺亚方舟在世界末日起航 科幻插画
```

![诺亚方舟在世界末日起航 科幻插画](demo/诺亚方舟在世界末日起航%20科幻插画.png)

```
时空 黑洞 辐射
```

![时空 黑洞 辐射](demo/时空%20黑洞%20辐射.png)

```
乡村 田野 屏保
```

![乡村 田野 屏保](demo/乡村%20田野%20屏保.png)

```
来自深渊 风景 绘画 写实风格
```

![来自深渊 风景 绘画 写实风格](demo/来自深渊%20风景%20绘画%20写实风格.png)
