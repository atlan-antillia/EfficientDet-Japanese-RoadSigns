# EfficientDet-Japanese-RoadSigns
Training and detection Japanese RoadSigns by EfficientDet

<h2>
EfficientDet Japanese RoadSigns (Updated: 2021/11/21)
</h2>

This is a simple python example to train and detect Japanese RoadSigns by EfficientDet of Google Brain AutoML.


<h3>
1. Installing tensorflow on Windows10
</h3>
We use Python 3.8 to run tensoflow 2 on Windows10.<br>
At first, you have to install Microsoft Visual Studio 2019 Community Edition for Windows10.<br>
We create and use "c:\google" folder for our project.<br>
<pre>
>mkdir c:\google
>cd    c:\google
>pip install - requirements.tx
>git clone https://github.com/cocodataset/cocoapi
>cd cocoapi/PythonAPI
</pre>
You have to modify extra_compiler_args in setup.py in the following way:<br>
   extra_compile_args=[],
<pre>
>python setup.py build_ext install
</pre>

<br>
<br>
<h3>
2. Installing EfficientDet-Japanese-RoadSigns
</h3>
We have merged our <b>EfficientDet-Japanese-RoadSigns</b> repository with <b>efficientdet</b> in 
<a href="https://github.com/google/automl">Google Brain AutoML</a> on 2021/09/12.<br>
Please clone EfficientDet-Japanese-RoadSigns in the working folder <b>c:\google</b>.<br>
<pre>
>git clone  https://github.com/atlan-antillia/EfficientDet-Japanese-RoadSigns.git<br>
</pre>
You can see the following folder <b>projects</b> in  EfficientDet-Japanese-RoadSigns folder of the working folder.<br>

<pre>
EfficientDet-Japanese-RoadSigns
└─projects
    ├─coco
    │  └─configs
    └─Japanese_RoadSigns
        ├─configs
        ├─outputs
        ├─saved_model
        │  └─variables
        ├─test
        ├─train
        ├─valid
        └─__pycache__
</pre>
<br>
<b>Note:</b><br>
 The Japanese_RoadSigns tfrecord in train and valid is created from the 
 <a href="https://github.com/atlan-antillia/YOLO_Annotated_Japanese_Roadsigns">YOLO_Annotated_Japanese_RoadSigns</a>
 However, the all images in the YOLO repository was resized to become 512x512 to create those japanese_roadsigns.tfrecords.<br>
<br>
 See also: <a href="https://github.com/atlan-antillia/YOLO_Annotated_Japanese_Roadsigns_512x512">YOLO_Annotated_Japanese_Roadsigns_512x512</a>
<br> 
<br>

<h3>
3. Viewer of tfrecord
</h3>
  If you would like to view each annotated image in a tfrecord, you can use  
<a href="https://github.com/jschw/tfrecord-view">tfrecord_view_gui.py</a>, or 
<a href="https://github.com/EricThomson/tfrecord-view">view_tfrecord_tf2.py</a>.
<br>
  We have created a new <i>modified_view_tfrecord_tf2.py</i> from the original <i>view_tfrecord_tf2.py</i> to be able 
  to accept a tfrecord file and label_map.pbtxt file on the command line.<br>
 
<pre>
python modified_view_tfrecord_tf2.py japanese_roadsigns.tfrecord label_map.pbtxt
</pre>
<br>
<br>
<table style="border: 1px solid #000;">

<tr><td><img src="./asset/tfrecord_1.png" width="512" height="auto"></td>
    <td><img src="./asset/tfrecord_2.png" width="512" height="auto"></td></tr>

<tr><td><img src="./asset/tfrecord_3.png" width="512" height="auto"></td>
    <td><img src="./asset/tfrecord_4.png" width="512" height="auto"></td></tr>

<tr><td><img src="./asset/tfrecord_5.png" width="512" height="auto"></td>
    <td><img src="./asset/tfrecord_6.png" width="512" height="auto"></td></tr>

<tr><td><img src="./asset/tfrecord_7.png" width="512" height="auto"></td>
    <td><img src="./asset/tfrecord_8.png" width="512" height="auto"></td></tr>
</table>
   
  
<h3>
4. Downloading the pretrained-model efficientdet-d0
</h3>
Please download an EfficientDet model chekcpoint file <b>efficientdet-d0.tar.gz</b>, and expand it in <b>EfficientDet-Japanese-RoadSigns</b> folder.<br>
<br>
https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
<br>
See: https://github.com/google/automl/tree/master/efficientdet<br>


<h3>
5. Training Japanese_RoadSigns by using pretrained-model
</h3>
We use the japanese_roadsigns_train.bat file.
Modified to use main2.py to write COCO metrics to the csv files(2021/11/21).
<pre>
>python main2.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./projects/Japanese_RoadSigns/train/japanese_roadsigns.tfrecord  ^
  --val_file_pattern=./projects/Japanese_RoadSigns/valid/japanese_roadsigns.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="label_map=./projects/Japanese_RoadSigns/configs/label_map.yaml" ^
  --model_dir=./projects/Japanese_RoadSigns/models ^
  --label_map_pbtxt=./projects/Japanese_RoadSigns/train/label_map.pbtxt ^
  --eval_dir=./projects/Japanese_RoadSigns/eval ^
  --ckpt=efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=4 ^
  --eval_samples=200  ^
  --num_examples_per_epoch=200 ^
  --num_epochs=100   


</pre>

<table style="border: 1px solid #000;">
<tr>
<td>
--mode</td><td>train_and_eval</td>
</tr>
<tr>
<td>
--train_file_pattern</td><td>./projects/Japanese_RoadSigns/train/japanese_roadsigns.tfrecord</td>
</tr>
<tr>
<td>
--val_file_pattern</td><td>./projects/Japanese_RoadSigns/valid/japanese_roadsigns.tfrecord</td>
</tr>
<tr>
<td>
--model_name</td><td>efficientdet-d0</td>
</tr>
<tr><td>
--hparams</td><td>"label_map=./projects/Japanese_RoadSigns/configs/label_map.yaml"
</td></tr>
<tr>
<td>
--model_dir</td><td>./projects/Japanese_RoadSigns/models</td>
</tr>
<tr><td>
--label_map_pbtxt</td><td>./projects/Japanese_RoadSigns/train/label_map.pbtxt
</td></tr>

<tr><td>
--eval_dir</td><td>./projects/Japanese_RoadSigns/eval
</td></tr>

<tr>
<td>
--ckpt</td><td>efficientdet-d0</td>
</tr>
<tr>
<td>
--train_batch_size</td><td>4</td>
</tr>
<tr>
<td>
--eval_batch_size</td><td>4</td>
</tr>
<tr>
<td>
--eval_samples</td><td>200</td>
</tr>
<tr>
<td>
--num_examples_per_epoch</td><td>100</td>
</tr>
<tr>
<td>
--num_epochs</td><td>100</td>
</tr>
</table>
<br>
<br>

<b>COCO meticss f and map at epoch41</b><br>
<img src="./asset/coco_metrics_epoch41.png" width="1024" height="auto">
<br>

<b>Train losses at epoch41</b><br>
<img src="./asset/train_losses_epoch41.png" width="1024" height="auto">
<br>

<b>COCO ap per class at epoch40</b><br>
<img src="./asset/coco_ap_per_class_epoch40.png" width="1024" height="auto">
<br>


<b>mAP at epoch 100</b><br>

<img src="./asset/mAP_epoch100.jpg" width="1024" height="auto">
<br>
<h3>
6. Create a saved_model from the checkpoint
</h3>
 We use japanese_roadsigns_create_saved_model.bat file.
<pre>
>python model_inspect.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./projects/Japanese_RoadSigns/models  ^
  --hparams="image_size=512x512" ^
  --saved_model_dir=./projects/Japanese_RoadSigns/saved_model

</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model</td>
</tr>

<tr>
<td>--model_name </td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--ckpt_path</td><td>./projects/Japanese_RoadSigns/models</td>
</tr>

<tr>
<td>--hparams</td><td>"image_size=512x512" </td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./projects/Japanese_RoadSigns/saved_model</td>
</tr>
</table>

<br>
<br>
<h3>
7. Detect japanese_road_signs by using a saved_model
</h3>
 We use japanese_roadsigns_detect.bat file.
<pre>
>mkdir .\projects\Japanese_RoadSigns\outputs
python model_inspect.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./projects/Japanese_RoadSigns/saved_model ^
  --min_score_thresh=0.3 ^
  --hparams=./projects/Japanese_RoadSigns/configs/detect.yaml ^
  --input_image=./projects/Japanese_RoadSigns/test/*.jpg ^
  --output_image_dir=./projects/Japanese_RoadSigns/outputs
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model_infer </td>
</tr>

<tr>
<td>--model_name</td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./projects/Japanese_RoadSigns/saved_model </td>
</tr>

<tr>
<td>--min_score_thresh</td><td>0.3 </td>
</tr>

<tr>
<td>--hparams</td><td>./projects/Japanese_RoadSigns/configs/detect.yaml </td>
</tr>

<tr>
<td>--input_image</td><td>./projects/Japanese_RoadSigns/test/*.jpg </td>
</tr>

<tr>
<td>--output_image_dir</td><td>./projects/Japanese_RoadSigns/outputs</td>
</tr>
</table>
<br>
<b>Fo details, see jpg files in the outputs folder:<b><br>

<img src="./asset/road_signs_outputs_2021-08-25_1.jpg" width="1280" height="auto"><br>
<br>
<br>
<img src="./asset/road_signs_outputs_2021-08-25_2.jpg" width="1280" height="auto"><br>

<br>
<br>
<h3>
8. Some detection results of Japanese RoadSigns
</h3>
<table style="border: 1px solid #000;">
<tr><td><img src="./projects/Japanese_RoadSigns/outputs/1.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/11.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/21.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/31.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/41.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/51.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/61.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/71.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/81.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/91.jpg" width="512" height="auto"></td></tr>


<tr><td><img src="./projects/Japanese_RoadSigns/outputs/101.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/111.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/121.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/131.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/141.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/151.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/161.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/171.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/181.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/191.jpg" width="512" height="auto"></td></tr>

<tr><td><img src="./projects/Japanese_RoadSigns/outputs/201.jpg" width="512" height="auto"></td>
    <td><img src="./projects/Japanese_RoadSigns/outputs/211.jpg" width="512" height="auto"></td></tr>

</table>
