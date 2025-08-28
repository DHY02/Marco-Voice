# 

#!/bin/bash
cd Models/marco_voice/

. ./path.sh || exit 1;

# Set paths for your custom data and pretrained models
custom_data_dir=/root/gpufree-data/Marco-Voice/data  # directory containing train, dev, test splits
processed_data_dir=/root/gpufree-data/Marco-Voice/processed  # processed data will be saved here
pretrained_model_dir=/root/gpufree-data/Marco-Voice/pretrained_models/CosyVoice-300M
exp_name=CosyVoice-300M-emo-2
trained_model_dir=/root/gpufree-data/Marco-Voice/trained_models/${exp_name}
# cp  ${custom_data_dir}/* ${processed_data_dir}/

# Define the stages you want to run.
# For custom data, you can skip the download stage.
stage=1
stop_stage=1
# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#   echo "Custom Data Preparation: Creating wav.scp, text, utt2spk, spk2utt files"
#   for split in train; do
#     mkdir -p ${processed_data_dir}/$split
#     # Assuming your custom data is organized as: ${custom_data_dir}/$split
#     python local/prepare_data.py --src_dir ${custom_data_dir}/ --des_dir ${processed_data_dir}/$split
#     echo "Processing done"
#   done
# fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract CampPlus Speaker Embeddings for Custom Data"
  for split in train; do
    python tools/extract_embedding_confidence_coefficient.py --dir ${custom_data_dir}/ \
      --onnx_path $pretrained_model_dir/campplus.onnx \
      --num_thread 10 || exit 1;
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract Discrete Speech Tokens for Custom Data"
  for split in train; do
    python tools/extract_speech_token.py --dir ${custom_data_dir}/ \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx || exit 1;
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Creating Parquet Format Data for Custom Data"
  for split in train; do
    mkdir -p ${processed_data_dir}/$split
    cp ${custom_data_dir}/* ${processed_data_dir}/$split
    mkdir -p ${processed_data_dir}/$split/parquet
    # cp ${processed_data_dir}/* ${processed_data_dir}/$split
    # cp ${processed_data_dir}/* ${processed_data_dir}/$split/parquet
    python tools/make_parquet_list_rodis.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir ${processed_data_dir}/$split \
      --des_dir ${processed_data_dir}/$split/parquet || exit 1;
  done
fi

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   echo "Running Inference on Custom Data"
#   for mode in sft zero_shot; do
#     python cosyvoice/bin/inference.py --mode $mode \
#       --gpu 0 \
#       --config conf/cosyvoice.yaml \
#       --prompt_data ${processed_data_dir}/test/parquet/data.list \
#       --prompt_utt2data ${processed_data_dir}/test/parquet/utt2data.list \
#       --tts_text `pwd`/tts_text.json \
#       --llm_model $pretrained_model_dir/llm.pt \
#       --flow_model $pretrained_model_dir/flow.pt \
#       --hifigan_model $pretrained_model_dir/hift.pt \
#       --result_dir `pwd`/exp/cosyvoice/test/$mode
#   done
# fi

export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Training on Normal Data"
  cat ${processed_data_dir}/train/parquet/data.list > ${processed_data_dir}/train.data.list
  for model in llm flow; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1232" \
      cosyvoice_rodis/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice_rodis_${model}.yaml \
      --train_data ${processed_data_dir}/train.data.list \
      --cv_data ${processed_data_dir}/train.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir "`pwd`/exp/cosyvoice/$model/${exp_name}/$train_engine" \
      --tensorboard_dir "`pwd`/tensorboard/cosyvoice/${exp_name}/$model/$train_engine" \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer || exit 1;
    done
fi

average_num=3
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  mkdir -p $trained_model_dir
  for model in llm flow; do
    decode_checkpoint=${trained_model_dir}/${model}.pt
    echo "Averaging Checkpoints for $model; final checkpoint: $decode_checkpoint"
    python cosyvoice_rodis/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path exp/cosyvoice/${model}/${exp_name}/torch_ddp  \
      --num ${average_num} || exit 1;
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Exporting Model for Inference"
  
  python cosyvoice_rodis/bin/export_jit.py --model_dir $trained_model_dir
  python cosyvoice_rodis/bin/export_onnx.py --model_dir $trained_model_dir
fi