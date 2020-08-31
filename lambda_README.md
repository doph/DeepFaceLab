# Guide


### Install

#### On any tweek instances

```
# config conda if it has not been done
source /ParkCounty/peDev/conda/etc/profile.d/conda.sh
conda init

# open a new terminal
conda activate dfl-lambda
```

#### Set up on your local machine 

You can use conda installation if your machine hasn't installed CUDA 10.0

```
conda create --name dfl-lambda python=3.6
conda activate dfl-lambda
conda install -n dfl-lambda cudatoolkit=10.0.130
conda install -n dfl-lambda cudnn=7.6.5=cuda10.0_0
conda install -n dfl-lambda pip

git clone https://github.com/lambdal/DeepFaceLab.git && \
cd DeepFaceLab && \
pip install -r requirements-cuda-tf1.15.3.txt
```

In your machine has CUDA 10.0 + CUDNN and you prefer virtualenv over conda, here is how to do it

```
git clone https://github.com/lambdal/DeepFaceLab.git && \
cd DeepFaceLab && \
virtualenv -p /usr/bin/python3.6 venv-tf-1.15.3 && \
. venv-tf-1.15.3/bin/activate && \
pip install -r requirements-cuda-tf1.15.3.txt
```

### Usage

####  TensorFlow trainer API

```
# AMP + TF1 API
# Simply pass both `use-amp` and `--api tf1`
# Recommended: To avoid OOM error during training highres + high dim model, 
# remove MatMul from TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST and add it to TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST
# Recommended: XLA can sometimes speed up mixed precision training by 20%

export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE=MatMul && \
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD=MatMul && \
export TF_XLA_FLAGS=--tf_xla_auto_jit=2 && \
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--use-amp \
--api tf1 \
--opt adam \
--lr 1e-05 \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name


# FP32 + TF1 API
# Simply drop `use-amp` and only pass `--api tf1`

export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE=MatMul && \
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD=MatMul && \
export TF_XLA_FLAGS=--tf_xla_auto_jit=2 && \
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--api tf1 \
--opt adam \
--lr 1e-05 \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name
```

* __Please Please use `clipgrad=True` when you use AMP__. Otherwise gradient will explode at some point.
* In practice, AMP achieves higher performance when __batch size and feature dimensions are multiples of 8__. So try to avoid number like `22`, use `16` or `24` instead.
* To avoid OOM error during training highres + high dim model, remove MatMul from `TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST` and add it to `TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST` (see the above example)
* [XLA](https://www.tensorflow.org/xla) is a compiler that can further increase mixed precision performance, as well as float32 performance to a lesser extent. Simply set the `TF_XLA_FLAGS` as shown in the above example. 
* Two optimizers are offered: `rmsprop` and `adam`. We set default optimizer to `rmsprop` to keep it consistent with DFL. However, we also offer `Adam` optimizer because it may converge faster.
* You can customize the initial learning rate `lr`. __We recommend `lr=1e-05` for adam optimizer, or `lr=5e-05`  for rmsprop optimizer.__
* __Reduce learning rate if you see loss increases / stuck at a high value (2.0)__. This is particularly useful for late stage of the training, and for contuning the training of a FP32 model in AMP.
* There are two types of GANs. `patch` (default) and `unetpatch` (a new DFL implementation added July 2020). You will be asked to choose between one of them after setting the `gan_power`. Words on the street is that `unetpatch` works better, but we haven't thoroughly tested it.
* User iteraction is the same as the original DFL, including using keyboard to control preview, save model etc. However, we close the preview window once trainig is finished, for the purpose of pipelining multi-stage training.
* Saved models can be loaded and re-trained by both APIs (`dfl` and `tf1`), and in both precisions (`fp32` and `amp`). 
* Only `SAEHD` model is supported. `Quick96` and `XSeg` can be added upon request.
* Learning rate dropout is not currently supported, and probably will not be supported. Learning rate decay should do a comparable, if not better, job in terms of optimizing the model at late stage of the training. 

####  DFL trainer API

To use the original DFL training API in FP32, simply do not pass `--api` so the default `dfl` API is used.

```
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name
```

* Only `rmsprop` optimizer is supported under the DFL training.
* The starting learning rate is configurable. To do so simply pass `--lr` to the command.
* Although you can use AMP with the DFL API, this is not recommended. The reason is that dynamic loss scaling is not supported by DFL due to its in-house implememation of optimizer.


### DeepVooDoo Model

We host experimental models in `models/Model_DeepVooDoo`. The current version is the same as the `SAEHD`, but offers the ability to customize the number of layers for the encoder, decoder, and inter blocks. 
* Simply pass `DeepVooDoo` as the model option to your `main.py` script, and you will be asked to set these hyper-parameters (`ae_scales`, `e_scales`) during the interacitve model configuration stage. 
* The scales for encoder and decoder are always kept the same so we only ask the users to set one of them (`e_scales`). By default `ae_scales = 1`, and `e_scales=4`. 
* Make sure `2^e_scales` are no larger than the image resolution. `ae_scales` does not have this problem as it is used for Dense Layers.
* `leaky_relu` layer is added between the Dense layers of the inter blocks. This hasn't been thouroughly tested but intuitivly makes sense, as it increases the non-linearly of the inter blocks so hopefully learns more complicated mappings.
* The settings will be printed in the model summary.
* Supported by both the `dfl` and the `tf1` training API, in both `FP32` and `AMP`

Example:

```
# AMP + TF1 API
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE=MatMul && \
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD=MatMul && \
export TF_XLA_FLAGS=--tf_xla_auto_jit=2 && \
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--use-amp \
--api tf1 \
--opt adam \
--lr 1e-05 \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model DeepVooDoo \
--force-gpu-idxs 0 \
--force-model-name your_model_name


# FP32 + TF1 API
# Simply drop `use-amp` and only pass `--api tf1`
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE=MatMul && \
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD=MatMul && \
export TF_XLA_FLAGS=--tf_xla_auto_jit=2 && \
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--api tf1 \
--opt adam \
--lr 1e-05 \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model DeepVooDoo \
--force-gpu-idxs 0 \
--force-model-name your_model_name


# FP32 + DFL API
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model DeepVooDoo \
--force-gpu-idxs 0 \
--force-model-name your_model_name
``` 


### Performance

#### FP32 v.s. AMP 

__1xQuadroRTX8000 training throughput (images/sec)__

|   | FP32  | AMP | AMP + XLA |
|---|---|---|---|
| SAEHD_liae_ud_gan_512_512_128_128_22, BS=4 | 0.83  | 1.66  | 2.10  |
| SAEHD_liae_ud_gan_512_256_128_128_32, BS=8 | 1.05  | 2.07 | 2.42 |
| SAEHD_liae_ud_512_256_128_128_32, BS=8 | 2.53  | 3.85 | 4.32 |

* Naming convention: `Model_resolution_ae_encoder_decoder_mask`
