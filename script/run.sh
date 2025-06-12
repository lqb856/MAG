#! /bin/bash

mkdir build
cd build
cmake ..
make

# 测试前修改！！
TAG="mag-bench2"

# 基础配置
BENCH_BIN="./test/test_mag"
LOG_DIR="/home/lqb485508/mips-ann-benchmark/logs"
OUTPUT_DIR="../output"
DATASET_BASE="/home/lqb485508/dataset"
RESULT_CSV="${LOG_DIR}/result1.csv"
LOG_FILE="${LOG_DIR}/benchmark_mag.log"

# 测试参数配置
declare -a DATASETS=(
    "TextToImage1M:200:Text-to-Image/base.1M.fbin:Text-to-Image/query.public.100K.fbin:Text-to-Image/gt_test_query100k_@100.bin"
    "Music100-1M:100:music/base_music100.fbin:music/query_music100.fbin:music/gt_music100_@100.bin"
)

# 测试前修改！！
declare -a INDEX_TYPES=(
    "efc400_r10_rip50_Rt5:Music100-1M:music/efc400_r10_rip50_Rt5_m.mips"
    "efc400_r20_rip40_Rt5:Music100-1M:music/efc400_r20_rip40_Rt5_m.mips"
    "efc400_r30_rip30_Rt5:Music100-1M:music/efc400_r30_rip30_Rt5_m.mips"
    "efc400_r40_rip20_Rt5:Music100-1M:music/efc400_r40_rip20_Rt5_m.mips"
    "efc400_r50_rip10_Rt5:Music100-1M:music/efc400_r50_rip10_Rt5_m.mips"
    "efc400_r10_rip50_Rt5:TextToImage1M:t2i/efc400_r10_rip50_Rt5_m.mips"
    "efc400_r20_rip40_Rt5:TextToImage1M:t2i/efc400_r20_rip40_Rt5_m.mips"
    "efc400_r30_rip30_Rt5:TextToImage1M:t2i/efc400_r30_rip30_Rt5_m.mips"
    "efc400_r40_rip20_Rt5:TextToImage1M:t2i/efc400_r40_rip20_Rt5_m.mips"
    "efc400_r50_rip10_Rt5:TextToImage1M:t2i/efc400_r50_rip10_Rt5_m.mips"
    # "psp-test:TextToImage1M:t2i/efc400-r40-a60-m10.mips"
)

# 动态参数范围
K=100

EF_VALUES=(100 200 400 1000 2000)
THREADS_VALUES=(32)

# 其他参数
LNN_VALUES=(0 2 4 10 20 30)

# 创建日志目录
mkdir -p ${LOG_DIR}

# 主测试循环
for dataset_info in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_name dim base_path query_path gt_path <<< "${dataset_info}"
    
    for index_info in "${INDEX_TYPES[@]}"; do
        IFS=':' read -r index_alias index_dataset index_path <<< "${index_info}"

        if [ "${dataset_name}" != "${index_dataset}" ]; then
            continue
        fi

        for ef in "${EF_VALUES[@]}"; do
            for nthreads in "${THREADS_VALUES[@]}"; do
                for lnn in "${LNN_VALUES[@]}"; do
                    # 构建命令参数
                    cmd=(
                        "${BENCH_BIN}"
                        "${DATASET_BASE}/${base_path}"
                        "${DATASET_BASE}/${query_path}"
                        "${OUTPUT_DIR}/${index_path}"
                        "${ef}"
                        "${K}"
                        "../output/result.txt"
                        "not_index"
                        "${dim}"
                        "${lnn}"
                        "${DATASET_BASE}/${gt_path}"
                        "${TAG}"
                        "${dataset_name}"
                        "${n_threads}"
                        "${RESULT_CSV}"
                        "${index_alias}_lnn${lnn}"
                    )

                    # 执行测试
                    echo "Running: ${cmd[@]}" | tee -a ${LOG_FILE}
                    time "${cmd[@]}" >> ${LOG_FILE} 2>&1

                    # 状态检查
                    if [ $? -eq 0 ]; then
                        echo "[SUCCESS] ${index_alias} completed" | tee -a ${LOG_FILE}
                    else
                        echo "[FAILED] ${index_alias}" | tee -a ${LOG_FILE}
                    fi
                done
            done
        done
    done
done

echo "All benchmarks completed. Results saved to ${LOG_DIR}"