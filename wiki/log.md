# CHIRINGUITO LOG

Pongamos aca lo que queremos hacer! [Markdown Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)

## TODOs
- [ ] Make again all kitti360 warpings without mirror
- [ ] Resnet50/Coco, but using the segmentation? I mean, RGB>SEGMENTED>fc512>LSTM
- [ ] CARLA + pix2pixHD https://github.com/NVIDIA/pix2pixHD

### DATASETS

- [ ] Make a new dataset of crossing types
  
    - [ ] Path planing: where to go with the car
    - [ ] Data recording (Stereo + Lidar + gopro) (¿Posicion?) *cameras with wide fov*
    - [ ] Labeling and corrections
    
- [ ] Kitti360 actualization

    - [ ] Warpings No mirror
    - [ ] Stereo + Lidar --> 3D-Mask 

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

### TRAIN LSTM 25.02.2021
Train of the lstm model respect to the resnet trained on **alcala-26.01.2021** (Same one used in the before test)

- [ ] Tested in pycharm with `python train.py --train --num_epochs 500 --wandb_group_id WORKSHOP.I21.sweep.lstm --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped/ --model LSTM --patience 5 --patience_start 25 --dataloader lstmDataloader_alcala26012021 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_de2u0bbr_30.pth --num_workers 8 --batch_size 32 --telegram --weighted --lstm_dropout 0.5 --fc_dropout 0.2 --lr 0.01 --scheduler --scheduler_type ReduceLROnPlateau`
- [ ] Creating sweep : wandb agent chiringuito/lstm-based-intersection-classficator/p3jbh0ln
