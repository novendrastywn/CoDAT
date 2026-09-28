# CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition

This is the official repository of 

[**CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition**](https://arxiv.org/abs/2608.06691)

*Novendra Setyawan, Chi-Chia Sun, Mao-Hsiu Hsu, Wen-Kai Kuo, Jing-Ming Guo, Jun-Wei Hsieh.* IEEE IoT Journal 2026

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Real-time human action recognition on Internet-of-Things (IoT) edge devices requires models that capture rich spatio-temporal cues within strict latency, memory, and power envelopes. Current 3D CNNs, video transformers, and shift-based ViT deliver high accuracy but come at computational costs that preclude edge IoT deployment. This paper proposes CoDAT, a Collaborative Dual-Attention Transformer that replaces conventional multi-head attention with a lightweight dual-branch module: Spatial Convolutional Attention (SCA) for local aggregation and Strided Single-Head Attention (SSHA) for global context. SSHA jointly compresses the spatial resolution and channel dimensions of the query, key, and value tensors via stride-based sparse projection, then fuses the resulting global and local features at a markedly reduced cost. To enable temporal communication across frames, a parameter-free TShift module is embedded in each block. Extensive experiments on Jetson AGX Orin and Raspberry Pi 5 demonstrate that CoDAT achieves an energy-accuracy balance in both image and action recognition. On ImageNet-1K, CoDAT-M runs 2x faster than EfficientViT384 and FastViT-S12 at comparable accuracy, and CoDAT-L matches ViT-S with 3x fewer parameters at 2x higher throughput. On Kinetics-400 and MA-52, CoDAT achieves competitive Top-1 accuracy against state-of-the-art CNN, transformer, and hybrid baselines while running up to 2.9x faster than VSwin-T, 2x faster than ViT-Temporal-Shift variants, and 5x faster than UniFormer-B. On UCF-101, CoDAT-S384 matches TokShift and LAPS while being 6x faster and requiring up to 13x fewer FLOPs, establishing an efficiency-accuracy balance for real-time action recognition in edge IoT perception systems..
</details>


## Pre-trained Models
| name | resolution | acc | #params | FLOPs | Throughput | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| CoDAT-S | 256x256 | 77.6 | 9.9M | 580M | 1439.6 | [model](https://huggingface.co/novendrastywn/CoDAT/resolve/main/codat_s1_inet_1k.pth) |
| CoDAT-M | 256x256 | 79.7 | 17.1M | 940M | 1021.9 | [model](https://huggingface.co/novendrastywn/CoDAT/resolve/main/codat_s2_inet_1k.pth) |
| CoDAT-L | 256x256 | 81.4 | 27.8M | 2.19G | 517.9 | [model](https://huggingface.co/novendrastywn/CoDAT/resolve/main/codat_s3_inet_1k.pth) |



## Training
### Image Classification

#### Setup
```bash
conda create -n codat python=3.11
conda activate shvit
conda install pytorch==2.8.0 torchvision==0.23.0 cudatoolkit=12.9 -c pytorch
pip install -r requirements.txt
```

#### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  validation/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

To train CoDAT models, follow the respective command below:
<details>
<summary>
CoDAT-S
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model codat_s --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.025
```
</details>

<details>
<summary>
CoDAT-M
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model codat_m --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.032
```
</details>

<details>
<summary>
CoDAT-L
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model codat_l --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.035
```
</details>



## Evaluation
Run the following command to evaluate a pre-trained CoDAT-S on ImageNet-1K validation set with a single GPU:
```bash
python main.py --eval --model codat_s --resume ./codat_s1.pth --data-path $PATH_TO_IMAGENET --input-size 256
```


## Latency Measurement
Run the following command to compare the throughputs on GPU/CPU:

```
python speed_test.py
```


## Citation
If our work or code help your work, please cite our paper:
```
@ARTICLE{11640850,
  author={Setyawan, Novendra and Sun, Chi-Chia and Hsu, Mao-Hsiu and Kuo, Wen-Kai and Guo, Jing-Ming and Hsieh, Jun-Wei},
  journal={IEEE Internet of Things Journal}, 
  title={CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Accuracy;Videos;Internet of Things;Energy;Transformers;Design methodology;Head;Costing;Costs;Vision transformer;dual attention;strided single-head attention;temporal shift;edge action recognition},
  doi={10.1109/JIOT.2026.3719793}}
```

## Acknowledgements
We sincerely appreciate [SHViT](https://github.com/ysj9909/SHViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT) and [PyTorch](https://github.com/pytorch/pytorch) for their wonderful implementations.
