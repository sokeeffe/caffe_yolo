#ifndef CAFFE_REGION_LOSS_LAYER_HPP_
#define CAFFE_REGION_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/region_layer.hpp"

namespace caffe {

template <typename Dtype>
class RegionLossLayer : public LossLayer<Dtype> {
public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "RegionWithLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal RegionLayer used to process conv output to box coords
  shared_ptr<Layer<Dtype> > region_layer_;
  // prob stores the output from the RegionLayer
  Blob<Dtype> prob_;
  // store delta for probs
  Blob<Dtype> delta_;
  // bottom vector holder used in call to underlying RegionLayer::Forward
  vector<Blob<Dtype>*> region_bottom_vec_;
  // top vector holder used in call to underlying RegionLayer::Forward
  vector<Blob<Dtype>*> region_top_vec_;

  int side_;
  int num_classes_;
  int coords_;
  int num_;
  float jitter_;

  float object_scale_;
  float class_scale_;
  float noobject_scale_;
  float coord_scale_;

  float thresh_;

  int iter_;

  vector<Dtype> biases_;
};

} //namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_