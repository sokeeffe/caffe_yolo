#include <algorithm>
#include <vector>

#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void RegionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  RegionParameter param = this->layer_param_.region_param();

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
void RegionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
}

template <typename Dtype>
void RegionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    //const Dtype alpha(1.0);
    //LOG(INFO) << "alpha:" << alpha;
    
  //   caffe_cpu_axpby(
  //       bottom[0]->count(),
  //       alpha,
  //       real_diff_.cpu_data(),
  //       Dtype(0),
  //       bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(RegionLayer);
#endif

INSTANTIATE_CLASS(RegionLayer);
REGISTER_LAYER_CLASS(Region);

}