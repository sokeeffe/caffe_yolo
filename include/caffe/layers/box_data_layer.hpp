#ifndef CAFFE_BOX_DATA_LAYER_HPP_
#define CAFFE_BOX_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/blob_writer.hpp"

namespace caffe {

template <typename Dtype>
class BoxDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BoxDataLayer(const LayerParameter& param);
  virtual ~BoxDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BoxData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  void Next();
  bool Skip();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
  int iter_;
};

}  // namespace caffe

#endif  // CAFFE_BOX_DATA_LAYER_HPP_
