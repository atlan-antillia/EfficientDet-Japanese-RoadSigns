python model_inspect.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./projects/Japanese_RoadSigns/models  ^
  --hparams="image_size=512x512" ^
  --saved_model_dir=./projects/Japanese_RoadSigns/saved_model
 