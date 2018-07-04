#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/layers/object_layer.hpp"
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
void ObjectLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ObjectParameter param = this->layer_param_.object_param();

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

  train_iter_ = 0;
  test_iter_ = 0;

  for (int c = 0; c < param.biases_size(); ++c) {
     biases_.push_back(param.biases(c)); 
  }
}

template <typename Dtype>
void ObjectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ObjectLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  // *********************DEBUG OBJECT INPUT************************************
  // char filename[200];
  // if (this->phase_ == TEST)
  //   sprintf(filename, "VerifyObject/object_bottom_output_test_%d_%d_%d_%d_%d.csv", test_iter_, top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
  // else
  //   sprintf(filename, "VerifyObject/object_bottom_output_train_%d_%d_%d_%d_%d.csv", train_iter_, top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
  // FILE *fp = fopen(filename, "w");
  // fp = fopen(filename, "w");
  // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  // for (int i = 0; i < bottom[0]->shape(1)*bottom[0]->shape(2); i++){
  //   int spatialSize = bottom[0]->shape(3);
  //   int j = i*spatialSize;
  //   for(; j < ((i+1)*spatialSize)-1;j++){
  //     fprintf(fp,"%f,",bottom_data[j]);
  //   }
  //   fprintf(fp,"%f\n",bottom_data[j]);
  // }
  // fflush(fp);
  // **********************END DUBUG OBJECT INPUT*********************************

  for (int i = 0; i < bottom[0]->count(); i++) {
    top_data[i] = sigmoid(top_data[i]);
  }
  if (this->phase_ == TEST)
    test_iter_++;
  else
    train_iter_++;
}

template <typename Dtype>
void ObjectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(ObjectLayer);
#endif

INSTANTIATE_CLASS(ObjectLayer);
REGISTER_LAYER_CLASS(Object);

}