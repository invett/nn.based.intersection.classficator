# CHIRINGUITO LOG

Pongamos aca lo que queremos hacer! [Markdown Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)

## TODOs
- [ ] Make again all kitti360 warpings without mirror
- [ ] Resnet50/Coco, but using the segmentation? I mean, RGB>SEGMENTED>fc512>LSTM

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
        - *out: todos bastante mal*

    - Pruebas RESNET50/COCO + average.pooling + FULLY CONNECTED
        - LR 0.005            out: mal, oscila                                    smoldering-candy-heart-432
        - LR 0.0001           out: un poco mejor, per todavia acc: ~ 0.18         handsome-lovebird-433
        - LR 0.000001         out: loss baja lento, pero igual queda a ~ 0.15     enthusiastic-lovebird-434
        - LR 0.01             out: como antes, ~ 0.17                             enthusiastic-lovebird-434

---

- TRAIN 15.02.2021
     
    Train RESNET+LSTM in two steps using alcala26:resnet and alcala12:lstm
  
    - resnet18
      - train.data + val.data = alcala-26.01.2021_selected_augmented_warped_1/
        - Difference with previous SWEEPS
            - set decimateStep = 2 (one of of two, 50%). This will differ from the previous SWEEPS!
            - using `26.01.2021_selected_augmented_warped_1` instead of `26.01.2021_selected_augmented_warped_5` of [Resnet Warping Alclaa](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/sweeps/n0r5wtvt)

        Tests:
        - [Kevinmusgrave](kevinmusgrave.github.io/pytorch-metric-learning/)
            - [x] [effervescent-hug-436](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/runs/27hiammi)
                `python train.py --batch_size=64 --dataloader alcala26012021 --decimate 2 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_augmented_warped_1 --distance_function pairwise --p 2 --lr 0.0005 --margin 5 --metric --miner --model resnet18 --normalize --num_epochs 2000 --num_workers 4 --optimizer adam --patience=5 --patience_start=50 --pretrained --telegram --train --wandb_group_id Resnet_Alcala_26 --weighted --nowandb`
            
            - [x] [compassionate-admirer-437](https://wandb.ai/chiringuito/lstm-based-intersection-classficator/runs/8qduw5og)
                `python train.py --batch_size=64 --dataloader alcala26012021 --decimate 2 --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_augmented_warped_1 --distance_function pairwise --p 2 --lr 0.005 --margin 5 --metric --miner --model resnet18 --normalize --num_epochs 2000 --num_workers 4 --optimizer adam --patience=5 --patience_start=50 --pretrained --telegram --train --wandb_group_id Resnet_Alcala_26 --weighted`

        - OUR metric learning
            - [ ]  []()
                `python train.py --train --embedding --batch_size=64 --lr=0.005 --optimizer adam --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_augmented_warped_1 --num_epoch 2000 --validation_step 5 --telegram --patience 2 --patience_start 50 --dataloader alcala26012021 --decimate 2 --model resnet18 --weighted --lossfunction MSE --centroids_path ./trainedmodels/teacher/all_embeddings_matrix.txt --wandb_group_id Resnet_Alcala_26 --nowandb`





