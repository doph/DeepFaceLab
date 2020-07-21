# Guide


### Prerequisite

- A machine with at least one GPU
- CUDA 10.0
- Python 3.6


### Install

Paste this command into a terminal to install Lambda's fork of DFL.

```
git clone https://github.com/lambdal/DeepFaceLab.git && \
cd DeepFaceLab && \
virtualenv -p /usr/bin/python3.6 venv-tf-1.15.3 && \
. venv-tf-1.15.3/bin/activate && \
pip install -r requirements-cuda-tf1.15.3.txt
```



### Usage

To use automatic mixed precision (AMP), simply pass `use-amp` to `train`. 

```
# Train in AMP

python main.py train \
--use-amp \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name


# Train in FP32
python main.py train \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name
```

The `use-amp` flag is also added to the end of the model summary:

```
============== Model Summary ===============
==                                        ==
==            Model name: test_SAEHD      ==
==                                        ==
==     Current iteration: 0               ==
==                                        ==
==------------ Model Options -------------==
==                                        ==
==            resolution: 512             ==
==             face_type: wf              ==
==     models_opt_on_gpu: False           ==
==                 archi: df              ==
==               ae_dims: 64              ==
==                e_dims: 64              ==
==                d_dims: 64              ==
==           d_mask_dims: 64              ==
==       masked_training: True            ==
==             eyes_prio: False           ==
==           uniform_yaw: True            ==
==            lr_dropout: y               ==
==           random_warp: False           ==
==             gan_power: 1.0             ==
==       true_face_power: 0.0             ==
==      face_style_power: 0.0             ==
==        bg_style_power: 0.0             ==
==               ct_mode: none            ==
==              clipgrad: True            ==
==              pretrain: False           ==
==       autobackup_hour: 0               ==
== write_preview_history: False           ==
==           target_iter: 100             ==
==           random_flip: False           ==
==            batch_size: 8               ==
==               use_amp: True            ==
==                                        ==
==-------------- Running On --------------==
==                                        ==
==          Device index: 0               ==
==                  Name: Quadro RTX 8000 ==
==                  VRAM: 47.46GB         ==
==                                        ==
============================================
```

### XLA



To acivate AMP, Simply add the `TF_XLA_FLAGS` environment variable when you run the program:

```
# XLA enabled

TF_XLA_FLAGS=--tf_xla_auto_jit=2 python main.py train \
--use-amp \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name
```


### Performance

- Model naming convention: `image resolution` _ `autoencoder dim` _ `encoder dim` _ `decoder dim`
- Memory usage is usually reduced by 50% by activating AMP.
- Here is an example comparison between training in [FP32](https://github.com/lambdal/DeepFaceLab/tree/master/logs/fp32.txt) and [AMP](https://github.com/lambdal/DeepFaceLab/tree/master/logs/amp.txt), with the same `SAEHD_liae_512_64_64_64` model.
- For small models, XLA can bring in an extra 10-15% speed up for FP32, and 30-50% speed up for AMP without accuracy loss. For middle/large models, XLA does not improve training throughput, and sometimes even performs worse.

#### FP32 v.s. AMP 

__1xTeslaV100-SXM3-32GB training throughput (images/sec)__

|   | FP32  | AMP | AMP + BS x2 |
|---|---|---|---|
| SAEHD_liae_128_128_64_64, BS=64 | 52.26  | 80.41 | 80.41 |
| SAEHD_liae_gan_128_128_64_64, BS=64 | 30.74  | 51.40  | 54.69 |
| SAEHD_liae_ud_gan_128_128_64_64, BS=64 | 44.79  | 66.72  | 75.56 |
| SAEHD_th_liae_ud_3_416_288_168_120, BS=8 | 1.37  | 2.61  | 3.29 |


#### XLA on v.s. off

__1xQuadroRTX8000-48GB training throughput (images/sec)__

|   | FP32  | FP32 + XLA  | AMP | AMP + XLA | AMP + BS x2 | AMP + BS x2 + XLA|
|---|---|---|---|---|---|---|
| SAEHD_liae_128_128_64_64, BS=64 | 40.53  | 45.03  | 55.27 | 75.99  | 63.99 | 90.07 | 
| SAEHD_liae_gan_128_128_64_64, BS=64 | 24.81  | 27.63 | 36.84  | 55.27 | 41.22 | 57.90 |


__1xTeslaV100-SXM3-32GB training throughput (images/sec)__

|   | FP32  | FP32 + XLA  | AMP | AMP + XLA | AMP + BS x2 | AMP + BS x2 + XLA|
|---|---|---|---|---|---|---|
| SAEHD_liae_ud_gan_256_128_64_64, BS=16 | 8.21 | 8.21 | 13.81 | 13.21 | 14.47| 14.82 |
| SAEHD_th_liae_ud_3_416_288_168_120, BS=8 | 1.46 | 1.42 | 2.92 | 2.26 |  3.41 | OOM |


