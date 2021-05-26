----------------
|RESNET VS LSTM|
----------------

between (..) parenthesis, the timestamp of txt files (also in filename) to identify the experiment.

------------------------------------------------------------------------------------------------------------------------

KITTI360-warped: RESNET.SVM + LSTM.FC

RESNET (1618487444): --test --metric --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt    --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader txt_dataloader      --model resnet18 --load_path ./trainedmodels/model_m5qcz4ha_75.pth --test_method svm --svm_mode Linear --telegram --nowandb --export_data
LSTM   (1618487481): --test          --dataset ../../DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt    --dataset_test ../../DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt --dataset_val ../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt --dataloader lstm_txt_dataloader --model LSTM     --lstm_hidden 32 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_m5qcz4ha_75.pth --load_path ./trainedmodels/model_n6tg093u_45.pth --telegram --nowandb --export_data

********

KITTI360_3D: RESNET.SVM + LSTM.FC

RESNET (1618485684): --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt        --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt                 --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt                       --dataloader txt_dataloader      --model resnet18 --load_path ./trainedmodels/model_y39tv127_35.pth --test_method svm --svm_mode Linear --telegram --nowandb --export_data
LSTM   (1618486638): --test          --dataset ../../DualBiSeNet/KITTI-360_3D/prefix_train_list.txt        --dataset_test ../../DualBiSeNet/KITTI-360_3D/prefix_test_list.txt                 --dataset_val ../../DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt                       --dataloader lstm_txt_dataloader --model LSTM     --lstm_hidden 256 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_y39tv127_35.pth --load_path ./trainedmodels/model_896m1nhe_45.pth --telegram --nowandb --export_data

********

KITTI360_3D_MASKED: RESNET.SVM + LSTM.FC

RESNET (1618487354): --test --metric --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt          --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt                --dataloader txt_dataloader      --model resnet18 --load_path ./trainedmodels/model_8juepfqw_50.pth --test_method svm --svm_mode Linear --telegram --nowandb --export_data
LSTM   (1618487402): --test          --dataset ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt --dataset_test ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt          --dataset_val ../../DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt                --dataloader lstm_txt_dataloader --model LSTM     --lstm_hidden 16 --lstm_layers 1 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_8juepfqw_50.pth --load_path ./trainedmodels/model_l6dext72_40.pth --telegram --nowandb --export_data


********

ALCALA26: RESNET.SVM + LSTM FC

RESNET (1619171509):  --test --metric --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped --dataloader txt_dataloader --model resnet18 --load_path ./trainedmodels/model_de2u0bbr_30.pth --test_method svm --svm_mode Linear --telegram --nowandb --export_data
LSTM   (1619172540):  --test --dataset ../../DualBiSeNet/alcala-26.01.2021_selected_warped --dataloader lstm_txt_dataloader --model LSTM --lstm_hidden 64 --lstm_layers 2 --feature_model resnet18 --feature_detector_path ./trainedmodels/model_de2u0bbr_30.pth --load_path ./trainedmodels/model_ey0sggz1_20.pth --telegram --nowandb --export_data