# CHIRINGUITO LOG

Pongamos aca lo que queremos hacer! [Markdown Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)

## TODOs
- [x] Make again all kitti360 warpings without mirror
- [ ] Training 'kevin' RESNET kitti360/Kitti2011 + TEST: SVM + Mahalanobis + FC (Pending fully conected)
- [x] Training LSTM with Kitti360/Kitti2011
- [x] Training LSTM with alcala26/alcala12
- [ ] Training LSTM 'Kevin' + TEST: SVM + Mahalanobis
- [ ] Temporal integration with Resnet Embeddings (¿Method to compare RESNET/LSTM? We can't just compare conf.matrices)
- [ ] Kitti360/Kitti2011 Segmentation masks
- [X] GAN: Add MASK fot the black part of the WARPING, as to not count those black pixels
- [X] GAN: use direct RGB image
- [X] GAN: Beef-up Generator -> 11 layers
- [X] GAN: Label smoothing strategy nn.BCEWithLogitsLoss(logits, labels*0.9) 
- [x] GAN: UNet Generator : update 12.03.2021 ... should be ok. need to test ; gave us artistic results...
- [X] GAN: PatchGAN Discriminator (Based on UNet Contracting Path)
- [X] GAN: Add MASK for the non-interesting parts of the RGB (eg. sky)  -> does not
- [ ] GAN: Pix2Pix to convert from road mask to RGB. (connected to 'futuristic wannabe'? see below)
- [X] GAN: StyleGAN2 (https://github.com/invett/stylegan2-pytorch/) /   
- [ ] GAN: StyleGAN2-ADA: Training Generative Adversarial Networks with Limited Data  https://github.com/NVlabs/stylegan2-ada-pytorch)
- [X] GAN: SWAGAN (https://github.com/invett/stylegan2-pytorch/)
- [ ] GAN: Conditional StyleGAN2 for balancing classes.
- [ ] GAN: Explore latent space -> Project images to latent space and find clusters -> Generate new geometries
- [ ] GAN: GauGAN -> https://blog.paperspace.com/gaugan-training-on-custom-datasets
- [ ] GRU vs LSTM: should be 1-line change

## FUTURISTIC wannabe
- [ ] CARLA + pix2pixHD https://github.com/NVIDIA/pix2pixHD

### DATASETS

- [ ] Make a new dataset of crossing types
  
    - [ ] Path planing: where to go with the car
    - [ ] Data recording (Stereo + Lidar + gopro) (¿Posicion?) *cameras with wide fov*
    - [ ] Labeling and corrections
    
- [ ] Kitti360 actualization

    - [x] Warpings No mirror --> warped
    - [x] Stereo + Lidar --> 3D                             ok 16.03.2021 
    - [ ] Stereo + Lidar + alvaromask --> 3D_masked 

- [ ] KITTI ROAD actualization

    - [x] Warpings No mirror --> warped
    - [x] Stereo + Lidar --> 3D                             ok 12.03.2021
    - [ ] Stereo + Lidar + alvaromask --> 3D_masked

## Log
### TRAIN 15.02.2021
   - teacher
        - RANDOM OSM
            - margin 10 + 750 elem en train, 150 en val.
            - margin  5 + 750 elem en train, 150 en val.
       
        - OSM de kitti:

    
*out: no parece para nada los viejos resultados de division de los clusters*

---

- **TRAIN** 15.02.2021: hablamos con javi.lo y 
  - LSTM
    
    - output de RESNET50/COCO + average.pooling    
        - [x] *out: todos bastante mal*

    - Pruebas RESNET50/COCO + average.pooling + FULLY CONNECTED
        - [x] LR 0.005            out: mal, oscila                                    smoldering-candy-heart-432
        - [x] LR 0.0001           out: un poco mejor, per todavia acc: ~ 0.18         handsome-lovebird-433
        - [x] LR 0.000001         out: loss baja lento, pero igual queda a ~ 0.15     enthusiastic-lovebird-434
        - [x] LR 0.01             out: como antes, ~ 0.17                             enthusiastic-lovebird-434

---

- TRAIN 15.02.2021
     
    Train RESNET+LSTM in two steps using alcala26:resnet and alcala12:lstm
  
    - resnet18
      - train.data + val.data = alcala-26.01.2021_selected_augmented_warped_1/
        - Difference with previous SWEEPS
            - set decimateStep = 2 (one of two, 50%). This will differ from the previous SWEEPS!
            - using `26.01.2021_selected_augmented_warped_1` instead of `26.01.2021_selected_augmented_warped_5` of [Resnet Warping Alclaa](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/sweeps/n0r5wtvt)

        Tests:
        - [Kevinmusgrave](kevinmusgrave.github.io/pytorch-metric-learning/)
            - [x] [effervescent-hug-436](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/runs/27hiammi)
                `python train.py --batch_size=64 --dataloader alcala26012021 --decimate 2 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_augmented_warped_1 --distance_function pairwise --p 2 --lr 0.0005 --margin 5 --metric --miner --model resnet18 --normalize --num_epochs 2000 --num_workers 4 --optimizer adam --patience=5 --patience_start=50 --pretrained --telegram --train --wandb_group_id Resnet_Alcala_26 --weighted --nowandb`
            
            - [x] [compassionate-admirer-437](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/runs/8qduw5og)
                `python train.py --batch_size=64 --dataloader alcala26012021 --decimate 2 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_augmented_warped_1 --distance_function pairwise --p 2 --lr 0.005 --margin 5 --metric --miner --model resnet18 --normalize --num_epochs 2000 --num_workers 4 --optimizer adam --patience=5 --patience_start=50 --pretrained --telegram --train --wandb_group_id Resnet_Alcala_26 --weighted`
              
            - [x] Prepare a sweep: [Sweep RESNET alcala26 - 3oqjl27d](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/sweeps/3oqjl27d)
                
        - OUR metric learning
            - [ ]  []()
                `python train.py --train --embedding --batch_size=64 --lr=0.005 --optimizer adam --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_augmented_warped_1 --num_epoch 2000 --validation_step 5 --telegram --patience 2 --patience_start 50 --dataloader alcala26012021 --decimate 2 --model resnet18 --weighted --lossfunction MSE --centroids_path ./trainedmodels/teacher/all_embeddings_matrix.txt --wandb_group_id Resnet_Alcala_26 --nowandb`



### TEST RESNET 25.02.2021 
Test of trained resnet trained on **alcala-26.01.2021** with SVM and MAHALANOBIS with respect to the following testing 
sets:

- `../../DualBiSeNet/alcala-26.01.2021_selected_warped/test/test_list.txt`
- `../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.120445AA.122302AA/seq.120445AA.122302AA.test_list.txt`
- `../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.164002AA.165810AA/seq.164002AA.165810AA.test_list.txt`

#### SVM
- [x] Lineal
    - [x] alcala-26
      `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode Linear --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-26.01.2021_selected_warped/test/test_list.txt --nowandb --metric=True --telegram --test=True`
    - [x] alcala-12 | 120445AA.122302AA | 000 sequence, focus, espartales
      `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode Linear --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.120445AA.122302AA/seq.120445AA.122302AA.test_list.txt --nowandb --metric=True --telegram --test=True`
    - [x] alcala-12 | 164002AA.165810AA | 000 sequence, focus, padres alvaro
      `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode Linear --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.164002AA.165810AA/seq.164002AA.165810AA.test_list.txt --nowandb --metric=True --telegram --test=True`
    
- [x] ovo
    - [x] alcala-26
      `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode ovo --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-26.01.2021_selected_warped/test/test_list.txt --nowandb --metric=True --telegram --test=True`
    - [x] alcala-12 | 120445AA.122302AA | 000 sequence, focus, espartales
      `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode ovo --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.120445AA.122302AA/seq.120445AA.122302AA.test_list.txt --nowandb --metric=True --telegram --test=True`
    - [x] alcala-12 | 164002AA.165810AA | 000 sequence, focus, padres alvaro
      `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode ovo --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.164002AA.165810AA/seq.164002AA.165810AA.test_list.txt --nowandb --metric=True --telegram --test=True`
    
#### Mahalanobis
- [x] alcala-26
  `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method mahalanobis --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-26.01.2021_selected_warped/test/test_list.txt --nowandb --metric=True --telegram --test=True`
- [x] alcala-12 | 120445AA.122302AA | 000 sequence, focus, espartales
  `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method mahalanobis --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.120445AA.122302AA/seq.120445AA.122302AA.test_list.txt --nowandb --metric=True --telegram --test=True`
- [x] alcala-12 | 164002AA.165810AA | 000 sequence, focus, padres alvaro
  `python train.py --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method mahalanobis --dataloader alcala26012021 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/train/train_list.txt --dataset_val ../../DualBiSeNet/alcala-26.01.2021_selected_warped/validation/validation_list.txt --dataset_test ../../DualBiSeNet/alcala-12.02.2021_warped/test.seq.164002AA.165810AA/seq.164002AA.165810AA.test_list.txt --nowandb --metric=True --telegram --test=True`

### TEST RESNET 02.03.2021 
Test of trained resnet trained on **Kitti360** with SVM and MAHALANOBIS with respect to the following testing 
sets:

- `../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt `
- `../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt `
- `../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt`

#### SVM
- [x] Lineal
    - [x] Kitti360
      `python train.py --test --load_path ./trainedmodels/model_12fuzxzd_55.pth --test_method svm --svm_mode Linear --dataloader alcala26012021 --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --nowandb --metric --telegram`
    - [x] Kitti2011
      `python train.py --test --load_path ./trainedmodels/model_12fuzxzd_55.pth --test_method svm --svm_mode Linear --dataloader alcala26012021 --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataset_test ../../DualBiSeNet/KITTI-ROAD_warped/prefix_all.txt --nowandb --metric --telegram`
    
- [x] ovo
    - [x] Kitti360
      `python train.py --test --load_path ./trainedmodels/model_12fuzxzd_55.pth --test_method svm --svm_mode ovo --dataloader alcala26012021 --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --nowandb --metric --telegram`
    - [x] Kitti2011
      `python train.py --test --load_path ./trainedmodels/model_12fuzxzd_55.pth --test_method svm --svm_mode ovo --dataloader alcala26012021 --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataset_test ../../DualBiSeNet/KITTI-ROAD_warped/prefix_all.txt --nowandb --metric --telegram`
    
#### Mahalanobis
- [x] Kitti360
  `python train.py --test --load_path ./trainedmodels/model_12fuzxzd_55.pth --test_method mahalanobis --dataloader alcala26012021 --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --nowandb --metric --telegram`
- [x] Kitti2011
  `python train.py --test --load_path ./trainedmodels/model_12fuzxzd_55.pth --test_method mahalanobis --dataloader alcala26012021 --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataset_test ../../DualBiSeNet/KITTI-ROAD_warped/prefix_all.txt --nowandb --metric --telegram`


### TRAIN LSTM 25.02.2021
Train of the lstm model respect to the resnet trained on **alcala-26.01.2021** (Same one used in the before test)

- [x] Tested in pycharm with 
  - SGD!!! `python train.py --train --num_epochs 500 --wandb_group_id WORKSHOP.I21.sweep.lstm --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/ --model LSTM --patience 5 --patience_start 25 --dataloader lstmDataloader_alcala26012021 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_de2u0bbr_30.pth --num_workers 8 --batch_size 32 --telegram --weighted --lstm_dropout 0.5 --fc_dropout 0.2 --lr 0.01 --scheduler --scheduler_type ReduceLROnPlateau`
  - ADAM: [con wandb](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/runs/2qmh4mfn) `python train.py --train --num_epochs 500 --wandb_group_id WORKSHOP.I21.sweep.lstm --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/ --model LSTM --patience 5--patience_start 25 --dataloader lstmDataloader_alcala26012021 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_de2u0bbr_30.pth --num_workers 4 --batch_size 16 --telegram --weighted --lstm_dropout 0.5 --fc_dropout 0.2 --lr 0.00001 --scheduler --scheduler_type ReduceLROnPlateau --all_in_ram True --optimizer adam` 
- [ ] Creating sweep muchos errores en esto, no entiendo porque. WARNING: adam needs much smaller LR, 0.01 sgd, 0.0001 for adam


### LSTM + Kevin
The idea is to exploit the trained RESNET, then the train is made as follows:
    - Select 3 sequences (to create a triplet) with different typologies, like: anchor (type-0), positive (type-0), negative (type-6)
    - Take the output of the LSTM (256 embedding size) and use the metrics (cosine/snr/etc) to create the loss 


### Trained models pth

- Alcala 26 (Resnet): model_de2u0bbr_30.pth
  `python train.py --TripletMarginMinerType=hard --batch_size=32 --dataloader=txt_dataloader --dataset=../../DualBiSeNet/alcala-26.01.2021_selected_warped --decimate=2 --distance_function=SNR --lr=7.019618311883417e-05 --margin=1.7381951948095211 --metric=true --miner=false --num_epochs=500 --num_workers=8 --optimizer=adamW --p=1 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep --weighted=true`
- Kitti Road (Resnet): model_hbrnonlu_10.pth
  `train.py --TripletMarginMinerType=hard --batch_size=64 --dataloader=txt_dataloader --dataset=../../DualBiSeNet/KITTI-ROAD_warped/train.prefix/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-ROAD_warped/test.prefix/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-ROAD_warped/validation.prefix/prefix_validation_list.txt --decimate=1 --distance_function=SNR --lr=5e-05 --margin=3.5 --metric=true --miner=false --num_epochs=500 --num_workers=8 --optimizer=adamW --p=1 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-ROAD --weight_tensor=Kitti2011 --weighted=true`
- Kitti 360 (Resnet): model_m5qcz4ha_75.pth
  `train.py --TripletMarginMinerType=hard --batch_size=32 --dataloader=txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --decimate=1 --distance_function=SNR --lr=0.00025 --margin=4 --metric=true --miner=false --num_epochs=500 --num_workers=8 --optimizer=adamW --p=1 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-360 --weight_tensor=Kitti360 --weighted=true`
- Kitti 360 3D (Resnet): model_xny9p5pw_75.pth
  `train.py --TripletMarginMinerType=all --batch_size=32 --dataloader=txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --decimate=2 --distance_function=SNR --lr=0.0006500000000000001 --margin=3.5 --metric=true --miner=true --num_epochs=500 --num_workers=4 --optimizer=adamW --p=2 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-360_3D --weight_tensor=Kitti360 --weighted=true`
- Kitti 360 3D-Masked (Resnet): model_8juepfqw_50.pth
  `train.py --TripletMarginMinerType=all --batch_size=32 --dataloader=txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --decimate=1 --distance_function=SNR --lr=0.008150000000000001 --margin=2 --metric=true --miner=false --num_epochs=500 --num_workers=4 --optimizer=adam --p=1 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-360_3D_Mask --weight_tensor=Kitti360 --weighted=true`
#########################  
- kitti Road (LSTM): model_04aysd3y_20.pth
  `train.py --batch_size=16 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-ROAD_warped/train.prefix/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-ROAD_warped/test.prefix/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-ROAD_warped/validation.prefix/prefix_validation_list.txt --decimate=2 --fc_dropout=0.2 --feature_detector_path=./trainedmodels/model_hbrnonlu_10.pth --feature_model=resnet18 --lossfunction=focal --lr=0.0008500000000000001 --lstm_dropout=0.4 --lstm_hidden=32 --lstm_layers=1 --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adam --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-ROAD.warpings.lstm --weight_tensor=Kitti2011 --weighted=false`
- kitti 360 (LSTM): model_n6tg093u_45.pth
  `train.py --batch_size=64 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --decimate=1 --fc_dropout=0.2 --feature_detector_path=./trainedmodels/model_m5qcz4ha_75.pth --feature_model=resnet18 --lossfunction=focal --lr=0.0011 --lstm_dropout=0.1 --lstm_hidden=32 --lstm_layers=1 --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adam --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-360.warpings.lstm --weight_tensor=Kitti360 --weighted=true`
- Alcala 26 (LSTM): (¿ERROR?)

- kitti 360 3D (LSTM): model_896m1nhe_45.pth
  `train.py --batch_size=64 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --decimate=1 --fc_dropout=0.1 --feature_detector_path=./trainedmodels/model_y39tv127_35.pth --feature_model=resnet18 --lossfunction=focal --lr=0.0019016929262780703 --lstm_dropout=0.1 --lstm_hidden=256 --lstm_layers=1 --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adam --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-360.3D.Masked.lstm --weight_tensor=Kitti360 --weighted=false`
- kitti 360 3D-Masked (LSTM): model_l6dext72_40.pth
  `train.py --batch_size=64 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --decimate=2 --fc_dropout=0.30000000000000004 --feature_detector_path=./trainedmodels/model_8juepfqw_50.pth --feature_model=resnet18 --lossfunction=focal --lr=0.0038430723524684953 --lstm_dropout=0.2 --lstm_hidden=16 --lstm_layers=1 --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adamW --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.KITTI-360.3D.Masked.lstm --weight_tensor=Kitti360 --weighted=true`
- kitti 360 (LSTM + Kevin): model_7ce52yge_70.pth
  `train.py --TripletMarginMinerType=hard --batch_size=16 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --decimate=1 --distance_function=pairwise --feature_detector_path=./trainedmodels/model_m5qcz4ha_75.pth --feature_model=resnet18 --lr=0.0022 --lstm_dropout=0.5 --lstm_hidden=256 --lstm_layers=1 --margin=1.5 --metric=true --miner=false --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adamW --p=2 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.Kitti360.warpings.lstm.metric --weight_tensor=Kitti360 --weighted=true`
- kitti 360 3D (LSTM + Kevin): model_74f2ypmx_15.pth
  `train.py --TripletMarginMinerType=all --batch_size=16 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --decimate=2 --distance_function=pairwise --feature_detector_path=./trainedmodels/model_y39tv127_35.pth --feature_model=resnet18 --lr=0.007544152629414655 --lstm_dropout=0.1 --lstm_hidden=256 --lstm_layers=2 --margin=1.5 --metric=true --miner=true --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adam --p=1 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.Kitti360-3D.warpings.lstm.metric --weight_tensor=Kitti360 --weighted=true`
- kitti 360 3D-Masked (LSTM + Kevin): model_fpggf2n0_30.pth
  `train.py --TripletMarginMinerType=hard --batch_size=16 --dataloader=lstm_txt_dataloader --dataset=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val=../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --decimate=2 --distance_function=pairwise --feature_detector_path=./trainedmodels/model_8juepfqw_50.pth --feature_model=resnet18 --lr=0.005904997194662464 --lstm_dropout=0.30000000000000004 --lstm_hidden=32 --lstm_layers=1 --margin=2 --metric=true --miner=false --model=LSTM --num_epochs=500 --num_workers=8 --optimizer=adam --p=1 --patience=5 --patience_start=30 --pretrained=true --scheduler=true --scheduler_type=ReduceLROnPlateau --telegram=true --train=true --wandb_group_id=WORKSHOP.IV21.sweep.Kitti360-3D_Masked.warpings.lstm.metric --weight_tensor=Kitti360 --weighted=true`

### Test trained models

- Alcala 26 (Resnet): model_de2u0bbr_30.pth
    - Linear SVM (93,66% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM (94,08% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode OVO --telegram`
    - Mahalanobis (94,99% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method mahalanobis --telegram`
      
- Kitti 360 (Resnet): model_m5qcz4ha_75.pth
    - Linear SVM (75,28% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_m5qcz4ha_75.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM (76,69% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_m5qcz4ha_75.pth --test_method svm --svm_mode ovo --telegram`
    - Mahalanobis (72,88% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_m5qcz4ha_75.pth --test_method mahalanobis --telegram`

- Kitti 360 3D (Resnet): model_y39tv127_35.pth
    - Linear SVM (77,96% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_y39tv127_35.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM (77,68% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_y39tv127_35.pth --test_method svm --svm_mode ovo --telegram`
    - Mahalanobis (73,16% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_y39tv127_35.pth --test_method mahalanobis --telegram`

- Kitti 360 3D-Masked (Resnet):  model_8juepfqw_50.pth
    - Linear SVM (77,68% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_8juepfqw_50.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM (76,69% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_8juepfqw_50.pth --test_method svm --svm_mode ovo --telegram`
    - Mahalanobis (73.02% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_8juepfqw_50.pth --test_method mahalanobis --telegram`

- Alcala 26 (LSTM): model_de2u0bbr_30.pth
    - Fully Connected:
      `python train.py --test --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 32 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_de2u0bbr_30.pth --load_path ./trainedmodels/model_1eg6hxj6_5.pth --telegram`
    - Linear SVM:
      
    - OVO SVM:
    
    - Mahalanobis:

- Kitti 360 (LSTM): model_m5qcz4ha_75.pth 
    - Fully Connected: model_n6tg093u_45.pth (76,59% ACC)
      `python train.py --test --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 32 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_m5qcz4ha_75.pth --load_path ./trainedmodels/model_n6tg093u_45.pth --telegram`
    - Linear SVM: model_7ce52yge_70.pth (74,46% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_m5qcz4ha_75.pth --load_path ./trainedmodels/model_7ce52yge_70.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM: model_7ce52yge_70.pth (78,72% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_m5qcz4ha_75.pth --load_path ./trainedmodels/model_7ce52yge_70.pth --test_method svm --svm_mode ovo --telegram`
    - Mahalanobis: model_7ce52yge_70.pth (72.91% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_m5qcz4ha_75.pth --load_path ./trainedmodels/model_7ce52yge_70.pth --test_method mahalanobis --telegram`

- Kitti 360 3D (LSTM): model_y39tv127_35.pth
    - Fully Connected: model_896m1nhe_45.pth (76,59% ACC)
      `python train.py --test --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_y39tv127_35.pth --load_path ./trainedmodels/model_896m1nhe_45.pth --telegram`
    - Linear SVM: model_74f2ypmx_15.pth (79,16% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 2 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_y39tv127_35.pth --load_path ./trainedmodels/model_74f2ypmx_15.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM: model_74f2ypmx_15.pth (79,16% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 2 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_y39tv127_35.pth --load_path ./trainedmodels/model_74f2ypmx_15.pth --test_method svm --svm_mode ovo --telegram`
    - Mahalanobis: model_74f2ypmx_15.pth (66,66% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 256 --lstm_layers 2 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_y39tv127_35.pth --load_path ./trainedmodels/model_74f2ypmx_15.pth --test_method mahalanobis --telegram`
      
- Kitti 360 3D-Masked (LSTM): model_8juepfqw_50.pth
    - Fully Connected: model_l6dext72_40.pth (82,97% ACC)
      `python train.py --test --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 16 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_8juepfqw_50.pth --load_path ./trainedmodels/model_l6dext72_40.pth --telegram`
    - Linear SVM: model_fpggf2n0_30.pth (85,10% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 32 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_8juepfqw_50.pth --load_path ./trainedmodels/model_fpggf2n0_30.pth --test_method svm --svm_mode Linear --telegram`
    - OVO SVM: model_fpggf2n0_30.pth (85,10% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 32 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_8juepfqw_50.pth --load_path ./trainedmodels/model_fpggf2n0_30.pth --test_method svm --svm_mode ovo --telegram`
    - Mahalanobis: model_fpggf2n0_30.pth (81,25% ACC)
      `python train.py --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 32 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_8juepfqw_50.pth --load_path ./trainedmodels/model_fpggf2n0_30.pth --test_method mahalanobis --telegram`


### GAN

 - Implementation of DCGAN (BCELoss) and WGAN (Wasserstein Loss)
 - Implementation of Conditional GAN (both BCELoss and WLoss)
 - Trials with warped images failed. Better performance with RGB images.
 - Lower bs and lower lr (linear relationship) better performance.
 - Add bias=False to convs and BN(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 - Change Generator and Discriminator architectures (More similar to DCGAN Pytorch)
 - Add MASK for the black part in WARPED  / MASK for RGB -> not working
 - Label smoothing strategy 
 - Trials with UNet Generator
 - Trials with PatchGAN Discriminator (UNet contraccting Path)
 - StyleGAN2 with Alcalá26 RGB's -> 40K iters 1d1h  ( run feasible-armadillo-5 )
 - SWAGAN with Alcalá26 + Alcalá12 RGB -> 40K iters 16h   ( run azure-disco-6 ) 
      - x2 FASTER!
      - More or less same quality, although some checkboard artifacts 
      - Same training curves, a bit worse PPL regularization
 - SWAGAN with Alcalá26 + Alcalá12 WARPED ( run visionary-flower-7 )
      - Improves StyleGAN2 PPL regularization from iter 30k
      - Add KITTI and RESUME from .pt iter40k (robust-snow-9)  ->  70k
 - Projector.py -> Explore latent  space
 - STyleGAN2 ADA with RGB
 - STyleGAN2 ADA with WARPED (Kitti + Alcalá ALL) 
