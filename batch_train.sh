#!/bin/bash
python3 train.py --batch_size=32 --lr=0.00001 --optimizer adam --dataset ../DualBiSeNet/data_raw_bev_mask/ --train --embedding --weighted --num_epoch 300 --validation_step 5 --telegram  --patience 2 --patience_start 50 --dataloader generatedDataset --lossfunction MSE --teacher_path ./trainedmodels/teacher/teacher_model_27e66514.pth
