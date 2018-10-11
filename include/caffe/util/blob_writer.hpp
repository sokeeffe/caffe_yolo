#ifndef CAFFE_UTIL_BLOB_WRITER_H_
#define CAFFE_UTIL_BLOB_WRITER_H_

#include <stdint.h>
#include "iostream"
#include "string"

#include "google/protobuf/message.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"


namespace caffe {

template <typename Dtype>
void WriteBlobToBinaryFile(const Blob<Dtype>& blob);

}   // namespace caffe

#endif  // CAFFE_UTIL_BLOB_WRITER_H_