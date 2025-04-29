#!/usr/bin/env bash

# Prepare submission
echo "exporting HF_TOKEN"
export HF_TOKEN=HF_TOKEN

export TRANSFORMERS_CACHE=/cache/path/

# huggingface-cli login      

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "exporting OPENAI_API_KEY"

export OPENAI_API_KEY="openai_api_key"


sleep 1

# rm -rf ./outputs/*
# echo "deleted previous results"

sleep 1

echo
echo "running program >>>>>>>>>>>>>>>>>>>"
echo


# python hf_apis.py
# python part_prompt_rebuilt.py

# python -u arxiv_lingua_sc.py --compress_rate 0.1 --start_point 263 --end_point 264 --evaluation_model qwen
# python -u gov_report.py --compress_rate 0.1 --start_point 1 --end_point 4 --evaluation_model qwen
# python -u multi_news.py --compress_rate 0.1 --start_point 1 --end_point 4 --evaluation_model qwen

# python -u hotpot_qa.py --compress_rate 0.2 --start_point 0 --end_point 3 --evaluation_model qwen

# python -u longbench_qa_eval.py --compress_rate 0.2 --start_point 0 --end_point 3 --evaluation_model qwen --dataset 2wikimqa_e

# python -u data_analysis_simplified.py --file ./outputs/arxiv_result_qwen_0.1_0to200_04272025_0012.json
# python -u data_analysis_simplified.py --file ./outputs/arxiv_result_qwen_0.3_0to200_04272025_0658.json
# python -u data_analysis_simplified.py --file ./outputs/arxiv_result_qwen_0.5_0to200_04272025_1532.json

# python -u data_analysis_simplified.py --file ./outputs/govreport_result_qwen_0.1_0to200_04272025_2037.json

# python -u data_analysis_simplified.py --file ./outputs/multinews_result_qwen_0.1_0to200_04282025_1031.json
# python -u data_analysis_simplified.py --file ./outputs/multinews_result_qwen_0.3_0to200_04282025_1457.json

# python -u data_analysis_simplified.py --file ./outputs/hotpot_qa_result_qwen_0.2_0to300_04272025_1542.json
# python -u data_analysis_simplified.py --file ./outputs/hotpot_qa_result_qwen_0.5_0to300_04272025_1614.json

# python -u data_analysis_simplified.py --file ./outputs/longbench_2wikimqa_e_result_qwen_0.2_0to300_04272025_1812.json
# python -u data_analysis_simplified.py --file ./outputs/longbench_2wikimqa_e_result_qwen_0.5_0to300_04272025_2104.json
