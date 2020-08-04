import os
import glob
import pandas as pd

list_gpu_type = ['QuadroRTX8000']
path_config = '/home/ubuntu/dfl-benchmark/output'
output_file = 'benchmark.csv'
list_config = [
'SAEHD_liae_ud_512_256_128_128_32', 'SAEHD_liae_ud_gan_512_256_128_128_32', 'SAEHD_liae_ud_256_128_64_64_32' \
]
list_gpu_idxs = ['0']


pattern = "]["

def get_time(t_start, t_end):
    t_s = [float(t) for t in t_start.split(':')]
    t_e = [float(t) for t in t_end.split(':')]

    if t_e[0] < t_s[0]:
        t_e[0] += 24

    sec_s = 3600 * t_s[0] + 60 * t_s[1] + t_s[2]
    sec_e = 3600 * t_e[0] + 60 * t_e[1] + t_e[2]
    return sec_e - sec_s 


def get_throughput(log_file_name):

    if not os.path.isfile(log_file_name):
        return 0.0
    count = 0
    time_second_iter = ''
    time_end = ''
    for i, line in enumerate(open(os.path.join(log_file_name))):
        if 'batch_size' in line:
            bs = line.split(' ')[13]
        if pattern in line:

            if count == 1:
                time_second_iter = line.split('][')[0][1:]
            count += 1
            time_end = line.split('][')[0][1:]
    t = get_time(time_second_iter, time_end) + 0.0001

    throughput = float(bs) * (count - 1) / t
    return throughput

list_row = []
for gpu_type in list_gpu_type:
    for gpu_idx in list_gpu_idxs:
        for config in list_config:
            num_gpu = len(gpu_idx.split(','))
            log_file = path_config + '/' + config + '_' + str(num_gpu) + 'x' + gpu_type + '*.txt'

            for log_file_name in glob.glob(log_file):
                items = os.path.basename(log_file_name).split('.')[0].split('_')
                
                name = "_".join(items[-3:])
                if not name in list_row:
                    list_row.append(name)
                    print(name)

df_throughput = pd.DataFrame(index=list_row, columns=list_config)

for gpu_type in list_gpu_type:
    for gpu_idx in list_gpu_idxs:
        for config in list_config:
            num_gpu = len(gpu_idx.split(','))
            log_file = path_config + '/' + config + '_' + str(num_gpu) + 'x' + gpu_type + '*.txt'
            for log_file_name in glob.glob(log_file):
                items = os.path.basename(log_file_name).split('.')[0].split('_')
                name = "_".join(items[-3:])
                print(log_file_name)
                try:
                    throughput = get_throughput(log_file_name)
                except:
                    throughput = 0.0
                df_throughput.at[name, config] = throughput

df_throughput.to_csv(output_file)
