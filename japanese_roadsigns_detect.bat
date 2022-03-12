python model_inspect.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./projects/Japanese_RoadSigns/saved_model ^
  --min_score_thresh=0.3 ^
  --hparams=./projects/Japanese_RoadSigns/configs/detect.yaml ^
  --input_image=./projects/Japanese_RoadSigns/test/*.jpg ^
  --output_image_dir=./projects/Japanese_RoadSigns/outputs