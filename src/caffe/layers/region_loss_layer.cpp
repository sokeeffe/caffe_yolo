#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

inline int entry_index(int side, int num_classes, int num, int coords, int batch, int location, int entry)
{
    int n =   location / (side*side);
    int loc = location % (side*side);
    return batch*side*side*num*(coords+num_classes+1) + n*side*side*(coords+num_classes+1) + entry*side*side + loc;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  RegionLossParameter param = this->layer_param_.region_loss_param();

  side_ = param.side();
  num_classes_ = param.num_classes();
  coords_ = param.coords();
  num_ = param.num();
  jitter_ = param.jitter();

  object_scale_ = param.object_scale();
  class_scale_ = param.class_scale();
  noobject_scale_ = param.noobject_scale();
  coord_scale_ = param.coord_scale();

  thresh_ = param.thresh();

  for (int c = 0; c < param.biases_size(); ++c) {
     biases_.push_back(param.biases(c)); 
  }
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  for (int b = 0; b < bottom[0]->num(); b++) {
    for (int n = 0; n < num_; n++) {
      int index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_, 0);
      for (int i = index; i < index+(side_*side_*2); ++i) {
        top_data[i] = sigmoid(top_data[i]);
      }
      index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_, coords_);
      for (int i = index; i < index+(side_*side_); ++i) {
        top_data[i] = sigmoid(top_data[i]);
      }
      index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_, coords_ + 1);
      for (int g = 0; g < side_*side_; ++g) {
        float sum = 0;
        float largest = -FLT_MAX;
        for (int i = 0; i < num_classes_; ++i) {
          if (top_data[index+g+(i*side_*side_)] > largest) largest = top_data[index+g+(i*side_*side_)];
        }
        for (int i = 0; i < num_classes_; ++i) {
          float e = exp(top_data[index+g+(i*side_*side_)] - largest);
          sum += e;
          top_data[index+g+(i*side_*side_)] = e;
        }
        for (int i = 0; i < num_classes_; ++i) {
          top_data[index+g+(i*side_*side_)] /= sum;
        }
      }
    }
  }
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(RegionLossLayer);
#endif

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}