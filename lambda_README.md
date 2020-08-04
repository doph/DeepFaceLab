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

* Two optimizers are offered: `rmsprop` and `adam`. We set default optimizer to `rmsprop` to keep consistent with DFL. However, we recommend `Adam` optimizer for both AMP and FP32 training because it converges faster. __So the rule of thumb is always use `--api tf1` together with `--opt adam`.__
* You can customize the initial learning rate `lr` and learning rate decay step `decay-step`. Precisely, the learning rate starts with the value of `lr`, then multiplied by `0.96` for every `decay-step`. __We recommend `lr=0.0001` if you train from scratch, and `lr=0.00001` to continue training at a late stage, or to train a model with GAN.__ Setting `decay-step` to 1000 seems reasonable for the current scale of the task.
* __Reduce learning rate if you see loss increases / stuck at a high value (2.0)__. This is particular useful for late stage of the training, and for contuning the training of a FP32 model in AMP.
* There are two types of GANs available. `patch` (default) and `unetpatch` (a new DFL implementation added July 2020). You will be asked to choose between one of them after setting the `gan_power`. Words on the street is that `unetpatch` works better, but we haven't throughly tested it.
* User iteraction is the same as the original DFL, including using keyboard to control preview, save model etc. However, we close the preview window once trainig is finished, for the purpose of pipelining multi-stage training.
* Saved models can be loaded and re-trained by both of APIs (`dfl` and `tf1`), and in both precisions (`fp32` and `amp`). 
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

|   | FP32  | AMP |
|---|---|---|
| SAEHD_liae_ud_gan_512_256_128_128_32, BS=8 | 1.05  | 2.07 |
| SAEHD_liae_ud_512_256_128_128_32, BS=8 | 2.53  | 3.85  |

* Naming convention: `Model_resolution_ae_encoder_decoder_mask`