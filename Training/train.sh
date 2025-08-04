# 

#!/bin/bash

model_type=$1
data_dir=$2
# 1. CosyVoice Model Training # bash run_training_rodis.sh for Rotational distance style modeling or bash run_training_emosphere.sh for  Spherical coordinate style modeling
# Step 1: Navigate to the CosyVoice Directory
# Open your terminal and change the working directory to the CosyVoice model location:

# if $model_type==cosyvoice:
# cd /mnt/workspace/user/code/MarcoVoice/Models/cosyvoice/
# # Step 2: Start the Training Script
# # Execute the provided training script to initiate the training process:
# bash run_training_emosphere.sh

# 2. F5-TTS Emotional Model Training
# # Note: If you aim to train an emotional model using F5-TTS, it's necessary to modify the __init__ file first.
# # Step 1: Enter the F5-TTS Directory
# # Move to the F5-TTS model project directory:
# cd /mnt/workspace/user/code/MarcoVoice/Models/F5-TTS
# # Step 2: Launch the Training
# # Utilize the accelerate library to start the training with the specified configuration:
# accelerate launch src/f5_tts/train/train.py --config-name F5TTS_Base_train.yaml

# # 3. Seamless M4T V2 Model Training
# # Step 1: Access the Seamless M4T V2 Directory
# # Change to the Seamless M4T V2 model directory:
# cd /mnt/workspace/user/code/Models/seamless_m4t_v2
# # Step 2: Begin the Training
# # Run the Python training script to start the model training:
# python training.py

# 1. CosyVoice Model Training
# bash run_training_rodis.sh for Rotational distance style modeling
# or bash run_training_emosphere.sh for Spherical coordinate style modeling

if [[ "$model_type" == "cosyvoice" ]]; then
    echo "Training CosyVoice model..."
    cd /mnt/workspace/user/code/MarcoVoice/Models/cosyvoice/ || exit
    bash run_training_emosphere.sh $data_dir
    cd -
elif [[ "$model_type" == "f5-tts" ]]; then
    echo "Training F5-TTS emotional model..."
    cd /mnt/workspace/user/code/MarcoVoice/Models/F5-TTS || exit
    accelerate launch src/f5_tts/train/train.py --config-name F5TTS_Base_train.yaml
    cd -
elif [[ "$model_type" == "seamless" ]]; then
    echo "Training Seamless M4T V2 model..."
    cd Marco-Voice/Models/seamless_m4t_v2 || exit
    python training.py
    cd -
else
    echo "Error: Invalid model type. Supported types are: cosyvoice, f5-tts, seamless"
    exit 1
fi