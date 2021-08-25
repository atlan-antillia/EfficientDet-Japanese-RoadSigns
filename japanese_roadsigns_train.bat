python main.py --mode=train_and_eval --train_file_pattern=./projects/Japanese_RoadSigns/train/japanese_roadsigns.tfrecord  --val_file_pattern=./projects/Japanese_RoadSigns/valid/japanese_roadsigns.tfrecord --model_name=efficientdet-d0  --model_dir=./projects/Japanese_RoadSigns/models --ckpt=efficientdet-d0  --train_batch_size=4 --eval_batch_size=4 --eval_samples=200  --num_examples_per_epoch=200 --num_epochs=100   
