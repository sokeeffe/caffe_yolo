// This program converts a set of images with object labels to a lmdb/leveldb
// by storing them as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should the folder where each file has the object and bounding box labels, in the format
//   class_num x_center_normalised y_center_normalised width_normalised height_normalised
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "jpg",
    "Optional: What type should we encode the image as ('png','jpg',...).");
//********************************* label_file **************************************//
DEFINE_string(label_file, "",
    "a map from name to label");
//********************************************************************************//

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and object labels to the\n"
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_yolo_data [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_yolo_data");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  //*************************************exit label_file*****************************************//
  const std::string label_file = FLAGS_label_file;
  if (label_file == "") {
    LOG(ERROR) << "empty label file";
    return 1;
  }
  //********************************************************************************//

  //******************************** produce label_file map***************************************//
  std::ifstream labelfile(label_file.c_str());
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(labelfile, line)) {
    size_t pos = line.find_last_of('/');
    lines.push_back(line.substr(pos+1)); 
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";
  //********************************************************************************//
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_image_folder(argv[1]);
  std::string root_label_folder(argv[2]);
  std::cout << "Root image folder: " << root_image_folder << '\n';
  std::cout << "Root label folder: " << root_label_folder << '\n';
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    // if (encoded && !enc.size()) {
    //   // Guess the encoding type from the file name
    //   string fn = lines[line_id].first;
    //   size_t p = fn.rfind('.');
    //   if ( p == fn.npos )
    //     LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
    //   enc = fn.substr(p);
    //   std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    // }
    size_t pos = lines[line_id].find_last_of('.');
    std::cout << "Image: " << root_image_folder+lines[line_id] << "\n";
    std::ifstream labelsfile((root_label_folder+lines[line_id].substr(0,pos)+".txt").c_str());
    std::vector<int> class_labels;
    std::vector<float> x_centers;
    std::vector<float> y_centers;
    std::vector<float> widths;
    std::vector<float> heights;
    // std::cout << "Label: " << (root_label_folder+lines[line_id].substr(0,pos)+".txt").c_str() << "\n";
    while (std::getline(labelsfile, line)) {
      // The label file structure needs to be "class x y w h"
      size_t next=0, last=0;
      next = line.find(' ',last);
      std::string label = line.substr(last, next-last);
      class_labels.push_back(atoi(label.c_str()));
      last = next+1;
      next = line.find(' ',last);
      std::string x_center = line.substr(last, next-last);
      x_centers.push_back(atof(x_center.c_str()));
      last = next+1;
      next = line.find(' ',last);
      std::string y_center = line.substr(last, next-last);
      y_centers.push_back(atof(y_center.c_str()));
      last = next+1;
      next = line.find(' ',last);
      std::string width = line.substr(last, next-last);
      widths.push_back(atof(width.c_str()));
      last = next+1;
      std::string height = line.substr(last);
      heights.push_back(atof(height.c_str()));
      // std::cout << "Class: " << label << " x: " << x_center << " y: " << y_center << " width: " << width << " height: " << height << "\n";
    }

    for (int i = class_labels.size(); i < 30; ++i){
      class_labels.push_back(-1);
      x_centers.push_back(0.);
      y_centers.push_back(0.);
      widths.push_back(0.);
      heights.push_back(0.);
    }

    //************************************************************************************//
    status = ReadBoxDataToDatum(root_image_folder + lines[line_id],
        class_labels, x_centers, y_centers, widths, heights,
        resize_height, resize_width, is_color, enc, &datum);
    //************************************************************************************//
    // break;
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id];

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
