# Perform object detection on NVIDIA Jetson with ONNX Runtime

1. Create a `Dockerfile` using the Jetson ONNX Runtime docker image and add application dependencies

    ```bash
    FROM mcr.microsoft.com/azureml/onnxruntime:v.1.4.0-jetpack4.4-l4t-base-r32.4.3
    WORKDIR .
    RUN apt-get update && apt-get install -y python3-pip libprotobuf-dev protobuf-compiler
    RUN python3 -m pip install onnx==1.6.0 easydict matplotlib
    CMD ["/bin/bash"]
    ```

2. Build a new image from the Dockerfile

    ```bash
    docker build -t jetson-onnxruntime-yolov4 .
    ```

3. Download the Yolov4 model, the object detection anchor locations, and class names from the ONNX model zoo

    ```bash
    wget https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx?raw=true -O yolov4.onnx
    wget https://raw.githubusercontent.com/onnx/models/master/vision/object_detection_segmentation/yolov4/dependencies/yolov4_anchors.txt
    wget https://raw.githubusercontent.com/natke/onnxruntime-jetson/master/coco.names
    ```

4. Download the Yolov4 object detection pre and post processing code

    ```bash
    wget https://raw.githubusercontent.com/natke/onnxruntime-jetson/master/preprocess_yolov4.py
    wget https://raw.githubusercontent.com/natke/onnxruntime-jetson/master/postprocess_yolov4.py
    ```

5. Download one or more test images

    ```bash
    wget https://raw.githubusercontent.com/SoloSynth1/tensorflow-yolov4/master/data/kite.jpg
    ```

6. Create an application called `main.py` to preprocess an image, run object detection and display the original image with the detected objects

    ```python
    import cv2
    import numpy as np
    import preprocess_yolov4 as pre
    import postprocess_yolov4 as post
    from PIL import Image

    input_size = 416

    original_image = cv2.imread("kite.jpg")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = pre.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    print("Preprocessed image shape:",image_data.shape) # shape of the preprocessed input        

    import onnxruntime as rt

    sess = rt.InferenceSession("yolov4.onnx")

    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name

    detections = sess.run([output_name], {input_name: image_data})[0]

    print("Output shape:", detections.shape)

    image = post.image_postprocess(original_image, input_size, detections)

    image = Image.fromarray(image)
    image.save("kite-with-objects.jpg")

7. Run the application

    ```bash
    docker run -it --rm -v $PWD:/workspace/ --workdir=/workspace/ --gpus all jetson-onnxruntime-yolov4 python3 main.py
    ```

The application reads in the kite image and locates all the objects in the image. You can try it with different images, and extend the application to use a video stream as shown in the Azure IoT edge application above.
