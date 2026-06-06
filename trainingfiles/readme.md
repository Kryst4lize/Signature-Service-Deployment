Order: 
1. stamp_augmentation.py     ← no direct run, it's a module (imported by step 2)

2. dataset_preparation.py    ← builds the CycleGAN dataset
   python dataset_preparation.py \
       --src   data/clean_signatures \
       --dst   data/cyclegan_dataset \
       --stamps stamp_noise

           Produces:
           data/cyclegan_dataset/
               trainA/   (clean)
               trainB/   (noisy)
               testA/
               testB/

3. [CycleGAN training]       ← external step, run your CycleGAN repo against
                                trainA+trainB / testA+testB
                                Output: a trained generator G(B→A)
                                        i.e.  noisy → clean

4. [Run G(B→A) on real docs] ← apply the trained CycleGAN generator to
                                your actual scanned documents to produce
                                cleaned signature crops

5. train_verification.py     ← train VGG16 + ResNet50 ON THE CLEANED crops
   python train_verification.py \
       --train_dir data/sign_data/train \
       --test_dir  data/sign_data/test  \
       --output    saved_models

           Produces:
           saved_models/
               vgg16_finetuned/
               vgg16_extractor/       ← FC1 output, 4096-d
               resnet50_finetuned/
               resnet50_extractor/    ← GAP output, 2048-d