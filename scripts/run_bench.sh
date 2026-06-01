#!/bin/bash

# MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=1 OUTPUT_LEN=1024 INPUT_LEN=1024     bash scripts/bench_deepseek_v4_flash.sh
# MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=2 OUTPUT_LEN=1024 INPUT_LEN=1024     bash scripts/bench_deepseek_v4_flash.sh
# MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=4 OUTPUT_LEN=1024 INPUT_LEN=1024     bash scripts/bench_deepseek_v4_flash.sh
# MODE=bench NUM_PROMPTS=10 MAX_CONCURRENT=8 OUTPUT_LEN=1024 INPUT_LEN=1024     bash scripts/bench_deepseek_v4_flash.sh


MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=1 OUTPUT_LEN=2048 INPUT_LEN=2048     bash scripts/bench_deepseek_v4_flash.sh
MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=2 OUTPUT_LEN=2048 INPUT_LEN=2048     bash scripts/bench_deepseek_v4_flash.sh
MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=4 OUTPUT_LEN=2048 INPUT_LEN=2048     bash scripts/bench_deepseek_v4_flash.sh
# MODE=bench NUM_PROMPTS=10 MAX_CONCURRENT=8 OUTPUT_LEN=2048 INPUT_LEN=2048     bash scripts/bench_deepseek_v4_flash.sh

MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=1 OUTPUT_LEN=4096 INPUT_LEN=4096     bash scripts/bench_deepseek_v4_flash.sh
MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=2 OUTPUT_LEN=4096 INPUT_LEN=4096     bash scripts/bench_deepseek_v4_flash.sh
MODE=bench NUM_WARMUPS=2 NUM_PROMPTS=10 MAX_CONCURRENT=4 OUTPUT_LEN=4096 INPUT_LEN=4096     bash scripts/bench_deepseek_v4_flash.sh
# MODE=bench NUM_PROMPTS=10 MAX_CONCURRENT=8 OUTPUT_LEN=4096 INPUT_LEN=4096     bash scripts/bench_deepseek_v4_flash.sh