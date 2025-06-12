#! /bin/bash

mkdir build
cd build
cmake ..
make

EFC=400
NEIGHBOR=60
THRESHOLD=5
N_L2=(10 20 30 40 50)

# 基础配置
BENCH_BIN="./test/test_mag"
LOG_DIR="/home/lqb485508/mips-ann-benchmark/logs"
OUTPUT_DIR="../output"
DATASET_BASE="/home/lqb485508/dataset"
RESULT_CSV="${LOG_DIR}/result1.csv"
LOG_FILE="${LOG_DIR}/build_mag.log"

# 测试参数配置
declare -a DATASETS=(
    "t2i:200:Text-to-Image/base.1M.fbin:Text-to-Image/query.public.100K.fbin:Text-to-Image/gt_test_query100k_@100.bin:/home/lqb485508/mips-ann-benchmark/output_binary/t2i/knn@400_ivf.bin"
    "music:100:music/base_music100.fbin:music/query_music100.fbin:music/gt_music100_@100.bin:/home/lqb485508/mips-ann-benchmark/output_binary/music/knn@400_ivf.bin"
)

if [ $? -eq 0 ]; then
  echo "Build successful"

  # 创建日志目录
  mkdir -p ${LOG_DIR}

  # 主测试循环
  for dataset_info in "${DATASETS[@]}"; do
      IFS=':' read -r dataset_name dim base_path query_path gt_path knn_path <<< "${dataset_info}"

      for R in "${N_L2[@]}"; do
          # 构建命令参数
          cmd=(
              "${BENCH_BIN}"
              "${DATASET_BASE}/${base_path}"
              "${knn_path}"
              "${EFC}"
              "${R}"
              "${EFC}"
              "${OUTPUT_DIR}/${dataset_name}/efc${EFC}_r${R}_rip$((NEIGHBOR - R))_Rt${THRESHOLD}_m${M}.mips"
              "index"
              "${dim}"
              "$((NEIGHBOR - R))"
              "0"
              "${THRESHOLD}"
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

  echo "All builds completed. Results saved to ${LOG_DIR}"
  
else
  echo "Build failed"
fi