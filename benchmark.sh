#!/bin/bash
OUTPUT_DIR=~/dfl-benchmark/output
MAIN_DIR=~/DeepFaceLab_dev
SRC_DIR=~/data/dfl/Snowden_face_small
DST_DIR=~/data/dfl/Gordon_face_small
GPUS=0
MODEL=SAEHD

# CONFIG=SAEHD_liae_ud_128_128_64_64_64
# BS_PER_GPU=128
# BS_PER_GPU2=256

# CONFIG=SAEHD_liae_ud_256_128_64_64_32
# BS_PER_GPU=32
# BS_PER_GPU2=64

CONFIG=SAEHD_liae_ud_512_256_128_128_32
BS_PER_GPU=8
# BS_PER_GPU2=16

# CONFIG=SAEHD_liae_ud_gan_512_256_128_128_32
# BS_PER_GPU=8

# CONFIG=SAEHD_liae_ud_512_512_128_128_22
# BS_PER_GPU=4

# CONFIG=SAEHD_liae_ud_gan_512_512_128_128_22
# BS_PER_GPU=4

MONITOR_INTERVAL=0.5

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader | sed 's/ //g' 2>/dev/null || echo PLACEHOLDER )"

IFS=', ' read -r -a gpus <<< "$GPUS"
NUM_GPU=${#gpus[@]}


run_benchmark() {
    model=$1
    config=$2
    setting=$3
    gpus=$4
    amp=$5
    api=$6
    opt=$7
    lr=$8
    bs_per_gpu=$9

    command_para="train \
    --training-data-src-dir=$SRC_DIR \
    --training-data-dst-dir=$DST_DIR \
    --config-file ${MAIN_DIR}/configs/${config}.yaml \
    --bs-per-gpu ${bs_per_gpu} \
    --force-model-name ${config} \
    --model-dir ${OUTPUT_DIR}/${GPU_NAME}/${config}"_"${setting} \
    --model $model \
    --force-gpu-idxs ${GPUS} \
    --api ${api} \
    --opt ${opt} \
    --lr ${lr} \
    --decay-step 200"

    if [ "$amp" == "on" ]; then
        command_para="${command_para} --use-amp"
    fi

    LOG_PATH=${OUTPUT_DIR}/${config}"_"${NUM_GPU}x${GPU_NAME}_bs${bs_per_gpu}"_"${setting}".txt"
    MONITOR_PATH=${OUTPUT_DIR}/${config}"_"${NUM_GPU}x${GPU_NAME}_bs${bs_per_gpu}"_"${setting}".csv"
    rm -rf ${OUTPUT_DIR}/${GPU_NAME}/${config}"_"${setting}
    flag_monitor=true
    echo $command_para 2>&1 | tee ${LOG_PATH}
    python3 ${MAIN_DIR}"/main.py" ${command_para} 2>&1 | tee -a $LOG_PATH &
    while $flag_monitor;
    do
        sleep $MONITOR_INTERVAL 
        last_line="$(tail -1 ${LOG_PATH})"
        if [[ $last_line == *"Done."* ]]; then
            flag_monitor=false
        else
            status="$(nvidia-smi -i $GPUS --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv)"
            echo "${status}" >> ${MONITOR_PATH}
        fi
    done

}

mkdir -p ${OUTPUT_DIR}/${GPU_NAME}

# run_benchmark $MODEL $CONFIG dfl-fp32 $GPUS off dfl rmsprop 0.00001 ${BS_PER_GPU}

# wait $! 
# run_benchmark $MODEL $CONFIG dfl-amp $GPUS on dfl rmsprop 0.00001 ${BS_PER_GPU}

# wait $! 
# run_benchmark $MODEL $CONFIG dfl-amp $GPUS on dfl rmsprop 0.00001 ${BS_PER_GPU2}

# wait $! 
# run_benchmark $MODEL $CONFIG tf1-fp32 $GPUS off tf1 adam 0.00001 ${BS_PER_GPU}

wait $! 
run_benchmark $MODEL $CONFIG tf1-amp $GPUS on tf1 adam 0.00001 ${BS_PER_GPU}

# wait $! 
# run_benchmark $MODEL $CONFIG tf1-amp $GPUS on tf1 adam 0.0001 ${BS_PER_GPU2}
