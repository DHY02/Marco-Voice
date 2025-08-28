#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
input_json=$1
output_path=$2
mkdir -p $output_path
# python Evaluation/ss_brainspeech.py --input_json ${input_json}  --output_txt ${output_path}/ss_speechbrain_text
# python Evaluation/ss_espnet.py  --input_json ${input_json} --output_txt ${output_path}/ss_espnet_text
# emotional acc
python Evaluation/emotion_acc.py ${input_json} ${output_path}/emotion_label.json
# # # #asr
# # conda activate tts-eval
# python Evaluation/asr.py --input_json ${input_json} --output_json ${output_path}/asr_output_json.json
# python Evaluation/calc_wer_non_latin.py --input_json ${output_path}/asr_output_json.json --output_txt ${output_path}/wer_txt
# cd Evaluation/DNS-Challenge/DNSMOS/
# python dns_mos.py --input_json ${input_json} --output_txt ${output_path}/dns_mos_txt

