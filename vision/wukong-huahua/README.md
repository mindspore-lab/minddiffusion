
# 悟空画画高性能版本

该分支为悟空画画高性能版本，目前仅有推理功能

## 快速使用

   1. 参考原始版本安装运行环境
   2. 下载Wukong-Huahua预训练权重文件 [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) 或移动微调后的权重文件至 wukong-huahua/models/ 目录.
   3. 运行指令可参考[examples/run_txt2img_infer.sh]

   ```python
      python txt2img.py --prompt "来自深渊 风景 绘画 写实风格"
   ```

## 性能对比

以下对比如无特殊说明均为910A上使用dpm-solver采样器采样15次的测试结果

|  代码版本 | batch size | image size  |  推理耗时 |
|  -  |  -  |  -  |  -  |
|  原版  |  4  |  512*512  |  9.8s   |
|  高性能版 |  4  |  512*512  |  4.2s   |
