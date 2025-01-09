#include <stdlib.h>

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class PrivacyProtector {
 public:
  PrivacyProtector(
      const std::string mode, int blur_size, int pixel_size,
      const std::string &mask_image_path,
      const std::string &model_path =
          "/mnt/d/Lab "
          "Practice/CPP/Project/models/face_detection_yunet_2023mar.onnx",
      const cv::Size &input_size = cv::Size(320, 320),
      float conf_threshold = 0.6f, float nms_threshold = 0.3f, int top_k = 5000,
      int backend_id = 0, int target_id = 0)
      : model_path_(model_path),
        input_size_(input_size),
        conf_threshold_(conf_threshold),
        nms_threshold_(nms_threshold),
        top_k_(top_k),
        backend_id_(backend_id),
        target_id_(target_id),
        mode(mode),
        blur_size(blur_size),
        pixel_size(pixel_size),
        mask_image_path(mask_image_path) {
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_,
                                       conf_threshold_, nms_threshold_, top_k_,
                                       backend_id_, target_id_);
    mask = cv::imread(mask_image_path, cv::IMREAD_UNCHANGED);
  }

  /* Overwrite the input size when creating the model. Size format: [Width,
   * Height].
   */
  void setInputSize(const cv::Size &input_size) {
    input_size_ = input_size;
    model->setInputSize(input_size_);
  }

  void detectFaces(const cv::Mat frame) {
    cv::Mat res;
    model->detect(frame, res);
    faces = res;
  }

  void applyBlur(cv::Mat face_region) {
    cv::GaussianBlur(face_region, face_region, cv::Size(blur_size, blur_size),
                     0);
  }

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

  void applyMask(cv::Mat face_region) {
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, face_region.size());
    resized_mask.copyTo(face_region, resized_mask);
  }

  void applyPrivacyProtection(cv::Mat &frame) {
    for (int i = 0; i < faces.rows; ++i) {
      // Get face region
      int x1 = static_cast<int>(faces.at<float>(i, 0));
      int y1 = static_cast<int>(faces.at<float>(i, 1));
      int w = static_cast<int>(faces.at<float>(i, 2));
      int h = static_cast<int>(faces.at<float>(i, 3));

      // Ensure the face region is within frame bounds
      if (x1 < 0 || y1 < 0 || x1 + w > frame.cols || y1 + h > frame.rows) {
        continue;  // Skip if the face region is out of bounds
      }
      cv::Mat face_region = frame(cv::Rect(x1, y1, w, h));

      // Apply the chosen privacy protection mode
      if (mode.compare("blur") == 0) {
        applyBlur(face_region);
      } else if (mode.compare("pixel") == 0) {
        applyPixelation(face_region);
      } else {
        applyMask(face_region);
      }
    }
  }

  cv::Mat visualize(const cv::Mat &image) {
    static cv::Scalar box_color{0, 255, 0};
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255, 0, 0),    // right eye
        cv::Scalar(0, 0, 255),    // left eye
        cv::Scalar(0, 255, 0),    // nose tip
        cv::Scalar(255, 0, 255),  // right mouth corner
        cv::Scalar(0, 255, 255)   // left mouth corner
    };
    static cv::Scalar text_color{0, 255, 0};

    auto output_image = image.clone();

    // Display current mode and parameters
    std::string mode_info = "Mode: " + mode;
    std::string param_info = "";

    // Add the relevant parameter value based on the current mode
    if (mode == "blur") {
      param_info = "Blur Size: " + std::to_string(blur_size);
    } else if (mode == "pixel") {
      param_info = "Pixel Size: " + std::to_string(pixel_size);
    }

    // Add the text at the top-left corner of the frame
    cv::putText(output_image, mode_info, cv::Point(10, 30),
                cv::FONT_HERSHEY_DUPLEX, 0.8, text_color, 2);
    cv::putText(output_image, param_info, cv::Point(10, 60),
                cv::FONT_HERSHEY_DUPLEX, 0.8, text_color, 2);

    for (int i = 0; i < faces.rows; ++i) {
      // Draw bounding boxes
      int x1 = static_cast<int>(faces.at<float>(i, 0));
      int y1 = static_cast<int>(faces.at<float>(i, 1));
      int w = static_cast<int>(faces.at<float>(i, 2));
      int h = static_cast<int>(faces.at<float>(i, 3));
      cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

      // Confidence as text
      float conf = faces.at<float>(i, 14);
      cv::putText(output_image, cv::format("%.4f", conf),
                  cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5,
                  text_color);

      // Draw landmarks
      for (int j = 0; j < landmark_color.size(); ++j) {
        int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)),
            y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
        cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
      }
    }
    return output_image;
  }

  void handleKeyboardInput(char key) {
    switch (key) {
      case 'q':
        std::cout << "Exiting..." << std::endl;
        exit(0);
      case '1':
        mode = "blur";
        std::cout << "Mode set to blur." << std::endl;
        break;
      case '2':
        mode = "pixel";
        std::cout << "Mode set to pixel." << std::endl;
        break;
      case '3':
        mode = "mask";
        std::cout << "Mode set to mask." << std::endl;
        break;
      case 'u':
        std::cout << "Enter new mask image path: " << std::endl;
        std::getline(std::cin, mask_image_path);
        mask = cv::imread(mask_image_path, cv::IMREAD_UNCHANGED);
        break;
      case '[':
        if (mode.compare("blur") == 0) {
          blur_size = std::max(1, blur_size - 20);
          std::cout << "Decrease blur size." << std::endl;
        } else if (mode.compare("pixel") == 0) {
          pixel_size = std::max(1, pixel_size - 3);
          std::cout << "Decrease pixel size." << std::endl;
        }
        break;
      case ']':
        if (mode.compare("blur") == 0) {
          blur_size += 20;
          std::cout << "Increase blur size." << std::endl;
        } else if (mode.compare("pixel") == 0) {
          pixel_size += 3;
          std::cout << "Increase pixel size." << std::endl;
        }
        break;
    }
  }

  void displayFrames(const cv::Mat frame) {
    auto res_image = visualize(frame);
    cv::imshow("Privacy Protector", res_image);
  }

  void start() {
    std::string address = "udp://127.0.0.1:5000?overrun_nonfatal=1";
    auto cap = cv::VideoCapture(address);
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    setInputSize(cv::Size(w, h));
    cv::Mat frame;
    while (true) {
      bool has_frame = cap.read(frame);
      if (!has_frame) {
        std::cout << "No frames grabbed! Exiting ...\n";
        break;
      }
      int key = cv::waitKey(1);
      if (key != -1) {
        handleKeyboardInput(static_cast<char>(key));
      }
      detectFaces(frame);
      applyPrivacyProtection(frame);
      displayFrames(frame);
    }
  }

 private:
  cv::Ptr<cv::FaceDetectorYN> model;
  std::string model_path_;
  cv::Size input_size_;
  float conf_threshold_;
  float nms_threshold_;
  int top_k_;
  int backend_id_;
  int target_id_;
  cv::Mat faces;
  cv::Mat mask;
  std::string mode;
  int blur_size;
  int pixel_size;
  std::string mask_image_path;
};

int main(int argc, char **argv) {
  cv::CommandLineParser parser(
      argc, argv,
      "{mode | blur | Set the initial mode}"
      "{blur_size | 15 | Set the initial blur kernel size}"
      "{pixel_size | 10 | Set the initial pixel block size}"
      "{mask_image | /mnt/d/Lab Practice/CPP/Project/images/Patrick.jpg | "
      "Specifie the path to the mask "
      "image}");

  std::string mode = parser.get<std::string>("mode");
  int blur_size = parser.get<int>("blur_size");
  int pixel_size = parser.get<int>("pixel_size");
  std::string mask_image_path = parser.get<std::string>("mask_image");

  PrivacyProtector model(mode, blur_size, pixel_size, mask_image_path);
  model.start();
  return 0;
}
