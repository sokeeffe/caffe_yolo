#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  //*********************************DEBUG iMAGE OUTPUT**********************************************************
  // LOG(INFO) << "String Length: " << strlen(filename.c_str())+1;
  // char *c = (char*)malloc(strlen(filename.c_str())+1);
  // strncpy(c, filename.c_str(), strlen(filename.c_str())+1);
  // char *next;
  // while((next = strchr(c,'/')))
  // {
  //   c = next+1;
  // }
  // char *copy = (char*)malloc(strlen(c)+1);
  // strncpy(copy, c, strlen(c)+1);
  // next = strchr(copy,'.');
  // if (next) *next=0;
  // char file_name[200];
  // sprintf(file_name, "VerifyValid_robby_ball_line0473/image_input_%s_%d_%d_%d.csv", copy, cv_img_origin.channels(),cv_img_origin.rows,cv_img_origin.cols);
  // FILE *fp = fopen(file_name, "w");
  // if(!fp) LOG(ERROR) << "Could not open or find file " << file_name;;
  // int i, j, k;
  // int num = 0;
  // // Iterate backwards over channels to convert from BGR to RGB
  // for (i = cv_img_origin.channels()-1; i >= 0; i--){
  //   for (j = 0; j < cv_img_origin.rows; j++) {
  //     for (k = 0; k < cv_img_origin.cols; k++) {
  //       // Values are stored in memory [rows,cols,channels] so convert to [channels,rows,cols]
  //       int index = j*cv_img_origin.channels()*cv_img_origin.cols + k*cv_img_origin.channels() + i;
  //       if (((num+1)%cv_img_origin.cols) != 0)
  //         fprintf(fp,"%f,",cv_img_origin.data[index]/255.0);
  //       else
  //         fprintf(fp,"%f\n",cv_img_origin.data[index]/255.0);
  //       num++;
  //     }
  //   }
  //     // int spatialSize = cv_img_origin.cols;
  //     // int j = i*spatialSize;
  //     // for (; j < ((i+1)*spatialSize)-1; j++){
  //     //     fprintf(fp,"%f,",cv_img_origin.data[j]/255.0);
  //     // }
  //     // fprintf(fp,"%f\n",cv_img_origin.data[j]/255.0);
  // }
  // fflush(fp);
  //*****************************END DEBUG IMAGE OUTPUT**************************************************
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
  } else {
    cv_img = cv_img_origin;
  }
  //****************************DEBUG RESIZED IMAGE OUT*************************************************
  // sprintf(file_name, "VerifyValid_robby_ball_line0473/resize_input_%s_%d_%d_%d.csv", copy, cv_img.channels(),cv_img.rows,cv_img.cols);
  // fp = fopen(file_name, "w");
  // if(!fp) LOG(ERROR) << "Could not open or find file " << file_name;;
  // num = 0;
  // // Iterate backwards over channels to convert from BGR to RGB
  // for (i = cv_img.channels()-1; i >= 0; i--){
  //   for (j = 0; j < cv_img.rows; j++) {
  //     for (k = 0; k < cv_img.cols; k++) {
  //       // Values are stored in memory [rows,cols,channels] so convert to [channels,rows,cols]
  //       int index = j*cv_img.channels()*cv_img.cols + k*cv_img.channels() + i;
  //       if (((num+1)%cv_img.cols) != 0)
  //         fprintf(fp,"%f,",cv_img.data[index]/255.0);
  //       else
  //         fprintf(fp,"%f\n",cv_img.data[index]/255.0);
  //       num++;
  //     }
  //   }
  // }
  // fflush(fp);
  //*************************END DEBUG RESIZED IMAGE OUT************************************************
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width,
    const bool is_color, const bool letterbox) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  // LOG(INFO) << "Sample: " << cv_img_origin.type();
  // LOG(INFO) << "Sample: " << cv_img_origin.at<uint8_t>(0,8,0);
  cv_img_origin.convertTo(cv_img_origin, CV_32F, 1.f/255);
  // LOG(INFO) << "FLOAT Sample: " << cv_img_origin.at<float>(0,8,0);
  // LOG(INFO) << "Sample: " << cv_img_origin.type();
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0 && !letterbox) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else if (height > 0 && width > 0 && letterbox) {
    int new_w = cv_img_origin.cols;
    int new_h = cv_img_origin.rows;
    if (((float)width/cv_img_origin.cols) < ((float)height/cv_img_origin.rows)) {
        new_w = width;
        new_h = (cv_img_origin.rows * width)/cv_img_origin.cols;
    } else {
        new_h = height;
        new_w = (cv_img_origin.cols * height)/cv_img_origin.rows;
    }
    cv::Mat cv_img_resized;
    cv::resize(cv_img_origin, cv_img_resized, cv::Size(new_w, new_h));
    // LOG(INFO) << "RESIZE Sample: " << cv_img_resized.at<float>(0,8,0);
    cv::Mat cv_img_back(cv::Size(width, height), cv_img_resized.type(), cv::Scalar(0.5));
    // LOG(INFO) << "BACKGROUND Sample: " << cv_img_back.at<float>(0,8,0);
    cv_img_resized.copyTo(cv_img_back(cv::Rect((width-new_w)/2,(height-new_h)/2,cv_img_resized.cols, cv_img_resized.rows)));
    cv_img = cv_img_back;
    // LOG(INFO) << "TOTAL Sample: " << cv_img.at<float>(8,8,0) << ","
    //           << cv_img.at<float>(50,50,0) << ","
    //           << cv_img.at<float>(100,100,0) << ","
    //           << cv_img.at<float>(150,150,0) << ","
    //           << cv_img.at<float>(200,200,0) << ","
    //           << cv_img.at<float>(280,280,0) << ",";
    cv_img.convertTo(cv_img, CV_32F, 255);
    cv_img.convertTo(cv_img, CV_8U);
    // LOG(INFO) << "TOTAL Sample: " << unsigned(cv_img.at<uint8_t>(8,8,0)) << ","
    //       << unsigned(cv_img.at<uint8_t>(50,50,0)) << ","
    //       << unsigned(cv_img.at<uint8_t>(100,100,0)) << ","
    //       << unsigned(cv_img.at<uint8_t>(150,150,0)) << ","
    //       << unsigned(cv_img.at<uint8_t>(200,200,0)) << ","
    //       << unsigned(cv_img.at<uint8_t>(280,280,0)) << ",";
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p+1) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

bool ReadBoxDataToDatum(const string& filename, const vector<int>& labels,
    const vector<float>& x_centers, const vector<float>& y_centers, const vector<float>& widths,
    const vector<float>& heights, const int height, const int width,
    const bool is_color, const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  // std::cout << "Rows: " << cv_img.rows << "\n";
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) ){
        std::cout << "CALLING HERE\n";
        return true;//ReadFileToDatum(filename, annoname, label_map, ori_w, ori_h, datum);
      }
      // std::cout << "Rows: " << cv_img.rows << "\n";
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_encoded(true);
      for (int i = 0; i < 30; ++i) {
        datum->add_float_data(float(labels[i]));
        datum->add_float_data(x_centers[i]);
        datum->add_float_data(y_centers[i]);
        datum->add_float_data(widths[i]);
        datum->add_float_data(heights[i]);
      }
      // read xml anno data
  //     ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
      return true;
    }
    // std::cout << "CALLING EVERYWHERE\n";
  //   CVMatToDatum(cv_img, datum);
  //   // read xml anno data
  //   ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV
}  // namespace caffe
