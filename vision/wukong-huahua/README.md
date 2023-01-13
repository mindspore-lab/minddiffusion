
# 悟空画画

## 目录

[Check English](./README_EN.md)

- [Wukong-Huahua悟空画画模型](#Wukong-Huahua悟空画画模型)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [准备checkpoint](#准备checkpoint)
  - [文图生成](#文图生成)
  - [训练微调](#训练微调)
  - [生成样例](#生成样例)

## Wukong-Huahua悟空画画模型

Wukong-Huahua是基于扩散模型的中文文生图大模型，由**华为诺亚团队**携手**中软分布式并行实验室**，**昇腾计算产品部**联合开发。模型基于[Wukong dataset](https://wukong-dataset.github.io/wukong-dataset/)训练，并使用[昇思框架(MindSpore)](https://www.mindspore.cn)+昇腾(Ascend)软硬件解决方案实现。
欢迎访问我们的[在线体验平台](https://xihe.mindspore.cn/modelzoo/wukong)试玩。

## 环境依赖

1. **昇腾软硬件解决方案(驱动+固件+CANN)**

   前往[昇腾社区](<https://www.hiascend.com/software/cann/commercial>)，按照说明下载安装。

2. AI框架 - **MindSpore** == 1.9

   前往[MindSpore官网](<https://www.mindspore.cn/install>)，按照说明下载安装。

   如需更多帮助，可以参考以下资料
   
   -  [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
   -  [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

3. **第三方依赖**

   ```python
   pip install -r requirements.txt
   ```

## 快速开始

### 准备checkpoint

下载Wukong-Huahua预训练参数 [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) 至 wukong-huahua/models/ 目录.

对于微调任务，我们提供了示例数据来展示格式，点击[这里](https://opt-release.obs.cn-central-221.ovaijisuan.com:443/wukonghuahua/dataset.tar.gz)下载.

### 文图生成

要进行文图生成，可以运行txt2img.py 或者直接使用默认参数运行 infer.sh.

```shell
python txt2img.py --prompt [input text] --ckpt_path [ckpt_path] --ckpt_name [ckpt_name] \
--H [image_height] --W [image_width] --output_path [image save folder] \
--n_samples [number of images to generate]
```
或者
```shell
bash scripts/infer.sh
```

更高的分辨率需要更大的显存. 对于 Ascend 910 芯片, 我们可以同时生成2张1024x768的图片或者16张512x512的图片。

### 训练微调

- 单卡微调

修改scripts/run_train.sh中相应配置

```shell
bash scripts/run_train.sh
```

- 多卡并行微调

修改scripts/run_train_parallel.sh中相应配置

```shell
bash scripts/run_train_parallel.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE]
```

### 生成样例

下面是我们的Wukong-Huahua模型生成的一些样例以及对应的`[input text]`。

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
