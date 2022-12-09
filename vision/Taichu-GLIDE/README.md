# Taichu-GLIDE
## 模型介绍
Taichu-GLIDE是**华为昇腾计算**携手**武汉人工智能研究院**、**中科院自动化所**基于昇腾昇思全栈开发的中文文生图大模型（紫东.太初系列模型之一），该模型采用了AIGC领域当前非常流行的扩散模型（Diffusion Model）技术，代码和预训练模型权重均对外进行开源，开发者可使用本仓进行以文生图任务的体验。


![一幅画着柯基的油画](https://user-images.githubusercontent.com/17930313/206085057-e079d90a-3313-4b9a-9e1c-f67a0594245d.png) 
**&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  一幅画着柯基的油画**

## 环境要求

1. **安装 CANN（5.1.RC2 版本）及其配套版本的驱动（driver）和 固件（firemware）**  \
    前往昇腾社区下载安装包：\
    <https://www.hiascend.com/software/cann/commercial> \
    以arm + 欧拉的系统配置为例 (x86的系统请选择x86的包) 

2. **安装 MindSpore 1.8.1 版本** \
    前往MindSpore官网，按照教程安装对应版本，链接如下: \
    <https://www.mindspore.cn/install>

3. **安装 requirements 依赖** \
    pip install -r requirements.txt

## 快速体验

### 推理
- 请先[点击此处](https://download.mindspore.cn/toolkits/minddiffusion/Taichu-GLIDE/)下载ckpt文件
- 在data/prompts.txt添加自己想要生成的prompt
- 修改 scripts/run_infer.sh中相关路径及配置
```bash
bash scripts/run_infer.sh
```
### 训练

```bash
# 生成阶段分布式训练
bash scripts/run_gen_finetune_dist.sh /path/hccl_xp_xxxx.json [DEVICE_NUM] [DEVICE_START]
```

```bash
# 超分阶段分布式训练
bash scripts/run_super_res_finetune_dist.sh /path/hccl_xp_xxxx.json [DEVICE_NUM] [DEVICE_START]
```
