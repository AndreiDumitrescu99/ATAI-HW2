python main.py --task 1 --dataset_path ./task1/train_data/annotations.csv --path_to_save_best_val_loss ./models/task1_pseoudolabel.pt --path_to_save_best_val_acc ./models/task1_best_acc_pseoudolabel.pt --pseudolabel True --path_to_unlabled_data ./task1/train_data/images/unlabeled/ --path_to_model_to_use_for_pseudo_label ./models/task1.pt