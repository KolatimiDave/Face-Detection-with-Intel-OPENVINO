<h2> Face-Detection-with-Intel-OPENVINO </h2>
Face-Detection Application demonstrates how to create a smart video IoT solution using Intel® hardware and software tools(OPENVINO). The app will detect faces of people in a designated area.


  | Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3|
| Time to Complete:    |  5 min     |

The DNN model used in this application is an Intel® optimized model that is part of the Intel® Distribution of OpenVINO™ toolkit. You can find it here:
```/opt/intel/openvino/deployment_tools/intel_models/face-detection-adas-0001```

<h3> Downloading Pretrained Model </h3>
Go to the **model downloader** directory present inside Intel® Distribution of OpenVINO™ toolkit:
Depending on your operating sysytem, the file path may be different.

  ```
  cd /opt/intel/openvino/deployment_tools/tools/model_downloader
  ```
Specify which model to download with `--name`.
- To download the face-detection-adas-0001, run the following command:

  ```
  sudo ./downloader.py --name face-detection-adas-0001
  ```
Specify which directory to store model's .xml and .bin file:
- To download model in a specific directory, run the following command:

  ```
  sudo ./downloader.py --name face-detection-adas-0001 -o /home/kolatimi/Desktop/Face-Detection-with-Intel-OPENVINO
  ```

<h3> Run the application </h3>
Go to people-counter-python directory:
```
cd <Face-Detection-with-Intel-OPENVINO_directory>
```
<h4> Setup the environment </h4>

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5 
```
<h3> Running on the CPU </h3>
```
python src/main.py -v cam -m ./intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -d CPU
```
<h3> Running on GPU </h3>
```
python src/main.py -v cam -m ./intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -d GPU
```
<br>
**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.<br>

<h3> Running on Intel® Neural Compute Stick </h3>
```
python src/main.py -v cam -m ./intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -d MYRIAD
```
