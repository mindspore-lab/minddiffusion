# Awesome Diffusion Models

## Introduction

This repo is an open source collection of a series of classic and new SoTA diffusion models including DDPM and stable diffusion based on MindSpore. Source codes and model weights are coming soon. 

## Model List
|Model|Paper|Institution|Date|Conference|Support|
|---|-------|------------|----|--------|----|
|DDPM|[DDPM: Denoising Difussion Probalisitic Model](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) |UC Berkeley|Jun 2020|NeurIPS 2020|To do|
|**Vision - Imege generation**||||||
|Improved diffusion|[Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a.html)|OpenAI|Feb 2021|PMLR 2021||
|Guided diffusion|[Diffusion Models Beat Gans on Image Synthesis](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)| OpenAI|Apr 2021|NeurIPS 2021||
|ADM|[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)|OpenAI|Apr 2021|NeurIPS 2021||
|FastDPM|[On Fast Sampling of Diffusion Probabilistic Models]()|NVIDIA|May 2021|ICLR Workshop 2021||
|LSGM|[Score-based Generative Modeling in Latent Space](https://arxiv.org/abs/2106.05931)|NVIDIA|Jun 2021|NeurIPS 2021||
|Distilled-DM|[Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)|Google Brain|Feb 2022|ICLR 2022|
|GGDM| [Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality](http://arxiv.org/abs/2202.05830)|Google Brain|Feb 2022|ICLR 2022|
|**Vision -  Text to Image**||||||
|Stable Diffusion/LDM| [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)| Stability.AI | Dec 2021|||
|Glide|[Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)|OpenAI|Dec 2021|||
|Dalle-2|[Hierarchical Text-conditional Image Generation with Clip Latents](https://arxiv.org/abs/2204.06125)|OpenAI|Apr 2022|||
|KNN Diffusion|[Image Generation via Large-Scale Retrieval](https://arxiv.org/abs/2204.02849)|Meta AI|Apr 2022|||
|Imagen|[Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)|Google Brain|May 2022|||
|LAION-RDM|[Text-Guided Synthesis of Artistic Images with Retrieval-Augmented Diffusion Models](https://arxiv.org/abs/2207.13038)|Ludwig-Maximilian University of Munich|Jul 2022|||
|DreamBooth|[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)|Google Research|Aug 2022|||
|DreamFusion|[DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)|Google Research|29 Sep 2022|||
|Vision - Image Editing||||||
|SDEdit|[SDEdit: Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)|Stanford U & CMU|Aug 2021|ICLR 2022||
|RePaint|[RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)|ETH Zurich|Jan 2022|CVPR 2022||
|Vision -  Video Genereation||||||
|Video diffusion models|[Video diffusion models](https://arxiv.org/abs/2204.03458)|Google Brain| Apr 2022|ICLR 2022 Workshop||
|MCVD|[MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853)|University of Montreal|May 2022||
|Make-A-Video|[Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792)|Meta AI| 29 Sep 2022|||
|Imagen Video|[Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303)|Google Brain|5 Oct 2022|||
|**Natural language**||||||
|Diffusion-LM|[Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)|Stanford University|May 2022|||
|**Audio - Audio Generation**|||||
|DiffWave|[DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761)|UCSD & Nvidia & Baidu|Jun 2020|ISMIR 2021||
|WaveGrad|[WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)|Google Brain|Sep 2020|ICLR 2021||
|Symbolic Music Generation|[Symbolic Music Generation with Diffusion Models](https://arxiv.org/abs/2103.16091)|Google Brain|Mar 2021|ISMIR 2021||
|DiffSinger|[DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)|Zhejiang University|May 2021|AAAI 2022||
|VDM|[Variational Diffusion Models](https://arxiv.org/abs/2107.00630)| Google Brain|Jul 2021|NeurIPS 2021||
|FastDiff|[FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis](https://arxiv.org/abs/2204.09934)|Tencent AI Lab|Apr 2022|IJCAI 2022||
|BDDMs|[BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis](https://arxiv.org/abs/2203.13508)|Tencent AI Lab|May 2022|ICLR 2022||
|SawSing|[DDSP-based Singing Vocoders: A New Subtractive-based Synthesizer and A Comprehensive Evaluation](https://arxiv.org/abs/2208.04756)||AUG 2022|ISMIR 2022||
|Prodiff|[ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech](https://arxiv.org/abs/2207.06389)| Zhejiang University|JUL 2022|ACM Muiltimedia 2022||
|**Audio -  Audio Conversion**|||||
|DiffVC|[Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme](https://arxiv.org/abs/2109.13821)|Huawei Noah|Sep 2021|ICLR 2022||
|**Audio -  Audio Enhancement**||||
|NU-Wave|[NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling](https://arxiv.org/abs/2104.02321),[💡](https://github.com/mindslab-ai/nuwave)|MINDSLAB & Seoul National University|Apr 2021|Interspeech 2021| 
|CDiffSE|[Conditional Diffusion Probabilistic Model for Speech Enhancement](https://arxiv.org/abs/2202.05256), [💡](https://github.com/neillu23/cdiffuse)|CMU|Feb 2022|IEEE 2022||
|**Audio -  Text to Speech**|||||
|Grad-TTS|[Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2111.11755)|Huawei Noah|May 2021||
|EdiTTS|[EdiTTS: Score-based Editing for Controllable Text-to-Speech](https://arxiv.org/abs/2110.02584.pdf), [💡](https://github.com/neosapience/EdiTTS)|Yale University|Oct 2021||
|DiffGAN-TTS|[DiffGAN-TTS: High-Fidelity and Efficient Text-to-Speech with Denoising Diffusion GANs](https://arxiv.org/abs/2201.11972),|Tencent AI Lab|Jan 2022||
|Diffsound| [Diffsound: Discrete Diffusion Model for Text-to-sound Generation](https://arxiv.org/abs/2207.09983), [💡](https://github.com/yangdongchao/text-to-sound-synthesis-demo)|Tencent AI Lab|Jul 2022||

## Contributing

We welcom all contributions to improve MindDiffusion! Please fork this repo and submit a pull request to contribute your diffusion models.
