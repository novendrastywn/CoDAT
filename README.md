# CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition

Official PyTorch implementation of **CoDAT**, published in the *IEEE Internet of Things Journal*.

> **CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition**
> Novendra Setyawan, Chi-Chia Sun, Mao-Hsiu Hsu, Wen-Kai Kuo, Jing-Ming Guo, Jun-Wei Hsieh
> IEEE Internet of Things Journal, 2026. [[Paper]](https://doi.org/10.1109/JIOT.2026.3719793)

<p align="center">
  <img src="CoDATFramework.png" width="90%" alt="CoDAT architecture"/>
</p>

## Highlights

- **CoDA (Collaborative Dual-Attention)** — a parallel dual-branch module combining:
  - **SSHA** (Strided Single-Head Attention): joint spatial and channel compression for global 2D attention at `(N²·Cv/sr⁴)` complexity
  - **SCA** (Spatial Convolutional Attention): local saliency gating at `(NC)` complexity, fused with SSHA through a learned projection
- **Low-cost temporal modeling** — a single post-CoDA **TShift** (temporal shift) per block: zero learnable parameters, temporal receptive field `2L+1` frames covers the full clip
- **Edge-first evaluation** — real hardware-measured latency and energy on **Jetson AGX Orin** (INA3221 via `jtop`) and **Raspberry Pi 5** (MXL7704 PMIC ADC via `vcgencmd pmic_read_adc`)

## Results

### Kinetics-400

| Model | Input (Frames×Res.) | Param (M) | GFLOPs | Top-1 (%) | Latency (ms/F)* | Energy (mJ/F)* |
|---|---|---|---|---|---|---|
| TSM-MobileNetV2 | 8×224² | 2.8 | 3.9 | 69.5 | 1.49 | 40.75 |
| TSM-R50 | 8×224² | 24.3 | 33.0 | 74.1 | 3.91 | 160.89 |
| MoViNet-A2 | 50×224² | 4.8 | 10.3 | 75.0 | 3.12 | 118.36 |
| SlowFast 8×8 | 32×256² | — | 65.7 | 77.0 | 7.05 | 286.33 |
| TokShift (ViT-B) | 8×224² | 85.9 | 135 | 77.3 | 10.15 | 511.35 |
| TimeSformer | 8×224² | 121.4 | 590 | 78.0 | 10.27 | 591.90 |
| UniFormer-S | 16×224² | 21.4 | 41.8 | 78.4 | 6.53 | 268.43 |
| VSwin-T | 32×224² | 28.0 | 88.0 | 78.8 | 6.96 | 365.20 |
| **CoDAT-S** | 8×256² | **10.6** | **4.6** | 73.6 | **0.89** | **13.79** |
| **CoDAT-M** | 8×256² | 17.7 | 7.5 | 75.3 | 1.43 | 37.14 |
| **CoDAT-M₃₈₄** | 8×384² | 17.7 | 16.8 | 77.5 | 2.56 | 86.8 |
| **CoDAT-L₃₈₄** | 8×384² | 28.4 | 39.1 | 78.5 | 4.74 | 198.7 |

CoDAT-M₃₈₄ matches ViT-Shift and outperforms TokShift at **3.6–3.9× lower latency** with 16.8 GFLOPs; CoDAT-L₃₈₄ reaches VSwin-T-level accuracy at roughly half its energy. Full comparison (including RPi5 measurements) in Table V of the paper.

### UCF-101 (8×256², single clip / single crop)

| Model | Pretrain | Param (M) | FLOPs (G) | Top-1 (%) | Latency (ms/F)* |
|---|---|---|---|---|---|
| TSM-MicroViT-S3 | IN-1K | 16.4 | 4.7 | 83.9 | 1.28 |
| TSM-EfficientViT-M5 | IN-1K | 12.2 | 4.3 | 86.1 | 1.56 |
| TSM-SHViT-S4 | IN-1K | 16.3 | 7.9 | 87.4 | 1.56 |
| **CoDAT-S** | IN-1K | **10.6** | **4.6** | **87.7** | **0.89** |
| **CoDAT-M** | IN-1K | 17.7 | 7.5 | **88.7** | 1.43 |
| TSM (ResNet-50) | K400 | 24.3 | 33.0 | 95.9 | 3.91 |
| TokShift | K400 | 85.9 | 135 | 95.4 | 10.15 |
| **CoDAT-S** | K400 | 10.6 | 4.6 | 94.2 | **0.89** |
| **CoDAT-M** | K400 | 17.7 | 7.5 | 95.2 | 1.43 |
| **CoDAT-S₃₈₄** | K400 | 10.6 | 10.3 | **95.4** | 1.60 |

\* Latency measured on Jetson AGX Orin with ONNX Runtime (CUDA EP), normalized per frame.

CoDAT-S₃₈₄ matches TokShift/LAPS accuracy at **~6× lower latency** and **~13× fewer FLOPs**. MA-52 and ImageNet-1K results are in the paper.

## Installation

```bash
git clone https://github.com/novendrastywn/CoDAT.git
cd CoDAT
conda create -n codat python=3.11 -y
conda activate codat
pip install torch torchvision timm onnxruntime pandas openpyxl
```

For edge benchmarking:

```bash
# Jetson AGX Orin
pip install jtop onnxruntime-gpu

# Raspberry Pi 5
pip install onnxruntime psutil
# power measurement uses the onboard PMIC ADC — no extra hardware needed:
vcgencmd pmic_read_adc
```


## Citation

```bibtex
@article{setyawan2026codat,
  title   = {CoDAT: Collaborative Dual-Attention Transformer with Low-Cost
             Temporal Modeling for Efficient Edge Action Recognition},
  author  = {Setyawan, Novendra and Sun, Chi-Chia and Hsu, Mao-Hsiu and
             Kuo, Wen-Kai and Guo, Jing-Ming and Hsieh, Jun-Wei},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  doi     = {10.1109/JIOT.2026.3719793}
}
```

## Acknowledgements

This work was supported by the National Science and Technology Council, Taiwan, under Grant NSTC-113-2221-E-305-018-MY3. The implementation builds upon [TSM](https://github.com/mit-han-lab/temporal-shift-module), [SHViT](https://github.com/ysj9909/SHViT), [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT), and [timm](https://github.com/huggingface/pytorch-image-models).

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
