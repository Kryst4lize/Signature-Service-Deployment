# RUN DOCKER
docker-compose up -d signature-minh 
# INTERACTIVE MODE
docker exec -it /bin/bash 
# RUN DOCKER COMPOSE 
docker compose run signature-minh
# BUILD DOCKER
docker build -t 10.254.144.152/tessel/training-service-base:0.0.1 -f Dockerfile .


# COMMAND RUN CYCLEGAN 
torchrun --nproc_per_node=2 train.py --dataroot ../data/cyclegan_processed_data --name signature --model cycle_gan --norm instance --continue_train --epoch 130 --epoch_count 131

# PREPARE DATASET CYCLEGAN
python dataset_preparation.py --src ../data/cyclegan_unprocessed_data --dst ../data/cyclegan_processed_data --stamps ../data/stamp_noise_data

# COMMAND RUN VERIFICATION
python train_verification.py  --train_dir ../data/verification_unprocessed_data/full --test_dir ../data/cyclegan_unprocessed_data/test  --output ../model/verification_model --backbone  both

# EVALUATE RUN VERIFICATION
python evaluate_verification.py \
    --test_dir  ../data/cyclegan_unprocessed_data/test \
    --output_dir ../model/evaluation