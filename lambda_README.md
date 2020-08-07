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
# We recommend setting the optimizer to adam, 
# and customize the initial learning rate and decay-step depending on the phase of the training job

# Optional: To avoid OOM error during training highres + high dim model, 
# remove MatMul from TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST and add it to TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE=MatMul
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD=MatMul

# Optional: XLA can sometimes speed up mixed precision training by 20%
export TF_XLA_FLAGS=--tf_xla_auto_jit=2

python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--use-amp \
--api tf1 \
--opt adam \
--lr 0.0001 \
--decay-step 1000 \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name


# FP32 + TF1 API
# Simply drop `use-amp` and only pass `--api tf1`

python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--api tf1 \
--opt adam \
--lr 0.0001 \
--decay-step 1000 \
--training-data-src-dir=your_src_dir \
--training-data-dst-dir=your_dst_dir \
--model-dir your_model_dir \
--model SAEHD \
--force-gpu-idxs 0 \
--force-model-name your_model_name
```

* __Please Please use `clipgrad=True` when you use AMP__. Otherwise gradient will explode at some point.
* To avoid OOM error during training highres + high dim model, remove MatMul from `TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST` and add it to `TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST` (see the above example)
* [XLA](https://www.tensorflow.org/xla) is a compiler that can further increase mixed precision performance, as well as float32 performance to a lesser extent. Simply set the `TF_XLA_FLAGS` as shown in the above example. 
* Two optimizers are offered: `rmsprop` and `adam`. We set default optimizer to `rmsprop` to keep it consistent with DFL. However, we recommend `Adam` optimizer for both AMP and FP32 training because it converges faster. __So the rule of thumb is always use `--api tf1` together with `--opt adam`.__
* You can customize the initial learning rate `lr` and learning rate decay step `decay-step`. Precisely, the learning rate starts with the value of `lr`, then multiplied by `0.96` for every `decay-step`. __We recommend `lr=0.0001` if you train from scratch, and `lr=0.00001` to continue training at a late stage, or to train a model with GAN.__ `--decay-step 1000` seems to be a reasonable choice for the scale of the tasks.
* __Reduce learning rate if you see loss increases / stuck at a high value (2.0)__. This is particularly useful for late stage of the training, and for contuning the training of a FP32 model in AMP.
* In practice, AMP achieves higher performance when __batch size and feature dimensions are multiples of 8__. So try to avoid number like `22`, use `16` or `24` instead.
* There are two types of GANs. `patch` (default) and `unetpatch` (a new DFL implementation added July 2020). You will be asked to choose between one of them after setting the `gan_power`. Words on the street is that `unetpatch` works better, but we haven't thoroughly tested it.
* User iteraction is the same as the original DFL, including using keyboard to control preview, save model etc. However, we close the preview window once trainig is finished, for the purpose of pipelining multi-stage training.
* Saved models can be loaded and re-trained by both APIs (`dfl` and `tf1`), and in both precisions (`fp32` and `amp`). 
* Only `SAEHD` model is supported. `Quick96` and `XSeg` can be added upon request.
* Learning rate dropout is not currently supported, and probably will not be supported. Learning rate decay should do a comparable, if not better, job in terms of optimizing the model at late stage of the training. 

####  DFL trainer API

To use the original DFL training API in FP32, simply do not pass `--api` so the default `dfl` API is used.

```
python3 /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DeepFaceLabAMP/main.py train \
--lr 0.0001 \
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
* Learning rate decay is currently not supported.



### Performance

#### FP32 v.s. AMP 

__1xQuadroRTX8000 training throughput (images/sec)__

|   | FP32  | AMP | AMP + XLA |
|---|---|---|---|
| SAEHD_liae_ud_gan_512_512_128_128_22, BS=4 | 0.83  | 1.66  | 2.10  |
| SAEHD_liae_ud_gan_512_256_128_128_32, BS=8 | 1.05  | 2.07 | 2.42 |
| SAEHD_liae_ud_512_256_128_128_32, BS=8 | 2.53  | 3.85 | 4.32 |

* Naming convention: `Model_resolution_ae_encoder_decoder_mask`
