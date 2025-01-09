## Real Time Privacy Protection Tool

### 1. Features

#### Mode 1: Blur
Press 1 to use this mode.
Applies a Gaussian blur to detected faces. The kernel size can be adjusted dynamically by pressing [ (decrease) or ] (increase) during runtime.

<img src=images/image.png width=60% />

#### Mode 2: Pixelation
Press 2 to use this mode.
Reduces the resolution of the face region and then scales it back up to create a pixelated effect. The block size can be adjusted dynamically by pressing [ (decrease) or ] (increase) during runtime.

<img src=images/image-1.png width=60% />

#### Mode 3: Mask
Press 3 to use this mode.
Covers the detected face region with a user-uploaded mask image. The mask is resized to fit the detected face region and can be replaced at runtime. If you want to do so, press U during the runtime to enter a new image path.

<img src=images/image-2.png width=60% />

### 2. Code Design
The code is structured in a single class, PrivacyProtector, which handles all functionalities related to face detection, privacy protection modes, and video frame processing. Key components of the class include:

#### 2.1 Face Detection (*detectFaces*)
The detectFaces function uses the YuNet face detection model from OpenCV's DNN module to detect faces in each frame. The bounding boxes of detected faces are stored in the faces matrix for further processing.
```cpp
void detectFaces(const cv::Mat frame) {
    cv::Mat res;
    model->detect(frame, res);
    faces = res;
}
```

#### 2.2 Privacy Protection (*applyPrivacyProtection*)
Based on the selected mode, the applyPrivacyProtection function applies one of the following privacy techniques:
```cpp
if (mode.compare("blur") == 0) {
    applyBlur(face_region);
} else if (mode.compare("pixel") == 0) {
    applyPixelation(face_region);
} else {
    applyMask(face_region);
}
```
- **Blur**: The applyBlur function applies a Gaussian blur to the detected face region.
```cpp
void applyBlur(cv::Mat face_region) {
    cv::GaussianBlur(face_region, face_region, cv::Size(blur_size, blur_size), 0);
}
```

- **Pixelation**: The applyPixelation function reduces the face region's resolution and then scales it back up to create a pixelated effect.
```cpp
void applyPixelation(cv::Mat face_region) {
    // Resize to a lower resolution (downscale)
    cv::Mat small_face;
    cv::resize(
        face_region, small_face,
        cv::Size(face_region.cols / pixel_size, face_region.rows / pixel_size),
        0, 0, cv::INTER_LINEAR);

    // Resize back to the original resolution (upscale)
    cv::resize(small_face, face_region, face_region.size(), 0, 0,
               cv::INTER_NEAREST);
}
```

- **Mask**: The applyMask function overlays a mask image over the detected face region.
```cpp
void applyMask(cv::Mat face_region) {
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, face_region.size());
    resized_mask.copyTo(face_region, resized_mask);
}
```

#### 2.3 Adjustment
Use **cv::waitKey()** to receive keyboard input while running the application. Handle the logic in *handleKeyboardInput()* function, including blur kernel size, pixel block size adjustment, mode switching, and mask image upload.
```cpp
int key = cv::waitKey(1);
if (key != -1) {
handleKeyboardInput(static_cast<char>(key));
}
```

#### 2.4 Command-Line Arguments
Use **cv::commandLineParser()** to parse the arguments.
```cpp
cv::CommandLineParser parser(
      argc, argv,
      "{mode | blur | Set the initial mode}"
      "{blur_size | 15 | Set the initial blur kernel size}"
      "{pixel_size | 10 | Set the initial pixel block size}"
      "{mask_image | /mnt/d/Lab Practice/CPP/Project/images/Patrick.jpg | "
      "Specifie the path to the mask "
      "image}");
```

### 3. Usage Instructions

#### 3.1 Video Capture
The project is developed under the Ubuntu system in WSL2. Developers cannot use the web camera directly in WSL2 due to historical reasons. Therefore, **FFmpeg** is needed to capture the video under Windows and then send the data to WSL2 via UDP.

To run the project, you need to download **FFmpeg** in both Windows and Ubuntu.

In windows, use root access to run the command in PowerShell:
```shell
ffmpeg -f dshow -i video="Integrated Camera" -preset ultrafast -vcodec libx264 -b:v 500k -f mpegts udp://<IP address>:<Port number>
```

Then the WebCam will be opened and the captured video stream will be sent to the Ubuntu, where main.cpp runs.

#### 3.2 Run the Program
Required CMake version: **3.10**

Required OpenCV version: **4.10**

In the root directory:
```shell
cd build
cmake ..
make
./private_protector
```

If you want to custmize the argument, please refer to the following example:
```shell
./private_protector -mode=mask -blur_size=10 -pixel_size=10 -mask_image=/path/to/image
```




