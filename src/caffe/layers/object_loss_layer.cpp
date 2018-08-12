#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/layers/object_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2)
{
    Dtype l1 = x1 - w1/2;
    Dtype l2 = x2 - w2/2;
    Dtype left = l1 > l2 ? l1 : l2;
    Dtype r1 = x1 + w1/2;
    Dtype r2 = x2 + w2/2;
    Dtype right = r1 < r2 ? r1 : r2;
    return right - left;
}

template <typename Dtype>
Dtype box_intersection(const vector<Dtype>& box, const vector<Dtype>& truth)
{
    Dtype w = overlap(box[0], box[2], truth[0], truth[2]);
    Dtype h = overlap(box[1], box[3], truth[1], truth[3]);
    if(w < 0 || h < 0) return 0;
    Dtype area = w*h;
    return area;
}

template <typename Dtype>
Dtype box_union(const vector<Dtype>& box, const vector<Dtype>& truth)
{
    Dtype i = box_intersection(box, truth);
    Dtype u = box[2]*box[3] + truth[2]*truth[3] - i;
    return u;
}

template <typename Dtype>
Dtype calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return box_intersection(box, truth)/box_union(box,  truth);
}

template <typename Dtype>
Dtype mag_array(Dtype* a, int n)
{
    int i;
    Dtype sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];   
    }
    return sqrt(sum);
}

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
vector<Dtype> get_cell_box(const Dtype* x, int i, int j, int w, int h) {
  vector<Dtype> b;
  b.clear();
  b.push_back((i+0.5)/(float)w);
  b.push_back((j+0.5)/(float)h);
  b.push_back(1.0/(float)w);
  b.push_back(1.0/(float)h);
  return b;
}

template <typename Dtype>
void delta_object_class(const Dtype* input_data, Dtype* delta, int index, int class_label, int classes, float scale, int stride, float* avg_cat)
{
  for (int n = 0; n < classes; ++n){
    delta[index + n*stride] = scale * (((n == class_label)?1 : 0) - input_data[index + n*stride]);
    //std::cout<<diff[index+n]<<",";
    if (n == class_label){
      *avg_cat += input_data[index + n*stride];
      //std::cout<<"avg_cat:"<<input_data[index+n]<<std::endl; 
    }
  }
}

template <typename Dtype>
void ObjectLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter object_param(this->layer_param_);
  object_param.set_type("Object");
  ObjectParameter param = this->layer_param_.object_param();
  object_layer_ = LayerRegistry<Dtype>::CreateLayer(object_param);
  object_bottom_vec_.clear();
  object_bottom_vec_.push_back(bottom[0]);
  object_top_vec_.clear();
  object_top_vec_.push_back(&prob_);
  object_layer_->SetUp(object_bottom_vec_, object_top_vec_);

  side_ = param.side();
  num_classes_ = param.num_classes();
  coords_ = param.coords();
  num_ = param.num();
  jitter_ = param.jitter();

  object_scale_ = param.object_scale();
  class_scale_ = param.class_scale();
  noobject_scale_ = param.noobject_scale();
  coord_scale_ = param.coord_scale();

  train_iter_ = 0;
  test_iter_ = 0;

  thresh_ = param.thresh();

  for (int c = 0; c < param.biases_size(); ++c) {
     biases_.push_back(param.biases(c)); 
  }
}

template <typename Dtype>
void ObjectLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  object_layer_->Reshape(object_bottom_vec_, object_top_vec_);
  delta_.ReshapeLike(prob_);
  // top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ObjectLossLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the object box coords and class prob values.
  object_layer_->Forward(object_bottom_vec_,object_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* delta_data = delta_.mutable_cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  caffe_set(delta_.count(), Dtype(0.0), delta_data);
  
  // ************************************DEBUG OBJECT OUTPUT LAYER****************************************
  // // LOG(INFO) << label_data[0] << "," << label_data[1] << "," << label_data[2] << ","
  // //       << label_data[3] << "," << label_data[4];
  // // LOG(INFO) << prob_data[0] << "," << prob_data[1] << "," << prob_data[2];
  // // LOG(INFO) << "Delta Shape: " << delta_.shape(0) << ","
  // //         << delta_.shape(1) << "," << delta_.shape(2) << ","
  // //         << delta_.shape(3);

  // char filename[200];
  // if (this->phase_ == TEST)
  //   sprintf(filename, "VerifyObject/object_output_test_%d_%d_%d_%d_%d.csv", test_iter_, prob_.shape(0), prob_.shape(1), prob_.shape(2), prob_.shape(3));
  // else
  //   sprintf(filename, "VerifyObject/object_output_train_%d_%d_%d_%d_%d.csv", train_iter_, prob_.shape(0), prob_.shape(1), prob_.shape(2), prob_.shape(3));
  // FILE *fp = fopen(filename, "w");
  // fp = fopen(filename, "w");
  // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  // for (int i = 0; i < prob_.shape(1)*prob_.shape(2); i++){
  //   int spatialSize = prob_.shape(3);
  //   int j = i*spatialSize;
  //   for(; j < ((i+1)*spatialSize)-1;j++){
  //     fprintf(fp,"%f,",prob_data[j]);
  //   }
  //   fprintf(fp,"%f\n",prob_data[j]);
  // }
  // fflush(fp);
  // ********************************END DEBUG OBJECT OUTPUT LAYER****************************************


  float avg_iou = 0;
  float recall = 0;
  float avg_cat = 0;
  float avg_obj = 0;
  float avg_anyobj = 0;
  int count = 0;
  int class_count = 0;
  for (int b = 0; b < bottom[0]->shape(0); b++) {
    for(int j = 0; j < side_; j++) {
      for(int i = 0; i < side_; i++) {
        vector<Dtype> pred;
        pred.clear();
        pred = get_cell_box(prob_data, i, j, side_, side_);
        // LOG(INFO) << "i: " << i << " j: " << j << " side: " << side_;
        // LOG(INFO) << "Box Pred: " << pred[0] << " " << pred[1] << " " << pred[2] << " " << pred[3];
        float best_iou = 0;
        for (int t = 0; t < 30; t++) {
          vector<Dtype> truth;
          truth.clear();
          Dtype x = label_data[b*30*5 + t*5 + 1];
          Dtype y = label_data[b*30*5 + t*5 + 2];
          Dtype w = label_data[b*30*5 + t*5 + 3];
          Dtype h = label_data[b*30*5 + t*5 + 4];

          if (!x) break;
          truth.push_back(x);
          truth.push_back(y);
          truth.push_back(w);
          truth.push_back(h);
          // LOG(INFO) << "Box Truth: " << truth[0] << " " << truth[1] << " " << truth[2] << " " << truth[3];
          Dtype iou = calc_iou(pred, truth);
          if(iou > best_iou) {
            best_iou = iou;
            // LOG(INFO) << "IOU: " << iou;
          }
        }
        int obj_index = b*side_*side_+j*side_+i;
        // int obj_index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_+j*side_+i, coords_);
          // avg_anyobj += prob_data[obj_index];
        if (best_iou > 0)
          delta_data[obj_index] = noobject_scale_ * (1 - prob_data[obj_index]);
        else
          delta_data[obj_index] = noobject_scale_ * (0 - prob_data[obj_index]);
      }
    }
  }
  Dtype loss = pow(mag_array(delta_data, delta_.count()),2);

  top[0]->mutable_cpu_data()[0] = loss;
  // LOG(INFO) << "Region Avg IOU: " << avg_iou/count << ", Class: " << avg_cat/class_count 
  //     << ", Obj: " << avg_obj/count << ", No Obj: " << avg_anyobj/(side_*side_*num_*bottom[0]->num())
  //     << ", Avg Recall: " << recall/count << ", count: " << count;
  // LOG(INFO) << "LOSS: " << loss;

  //*****************************************DEBUG REGION LOSS***************************************************************
  // if(this->phase_ == TEST)
  //   sprintf(filename, "VerifyObject/object_delta_test_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // else
  //   sprintf(filename, "VerifyObject/object_delta_train_%d_%d_%d_%d_%d.csv", train_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // fp = fopen(filename, "w");
  // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  // for (int i = 0; i < delta_.shape(1)*delta_.shape(2); i++){
  //   int spatialSize = delta_.shape(3);
  //   int j = i*spatialSize;
  //   for(; j < ((i+1)*spatialSize)-1;j++){
  //     fprintf(fp,"%f,",delta_data[j]);
  //   }
  //   fprintf(fp,"%f\n",delta_data[j]);
  // }
  // fflush(fp);

  // if(this->phase_ == TEST)
  //   sprintf(filename, "VerifyObject/object_loss_test_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // else
  //   sprintf(filename, "VerifyObject/object_loss_train_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // fp = fopen(filename, "w");
  // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  // fprintf(fp,"%f,",loss);
  // fflush(fp);

  // if(this->phase_ == TEST)
  //   sprintf(filename, "VerifyObject/object_label_test_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // else
  //   sprintf(filename, "VerifyObject/object_label_train_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // fp = fopen(filename, "w");
  // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  // for(int j = 0; j < side_; j++) {
  //   for(int i = 0; i < side_; i++) {
  //     vector<Dtype> pred;
  //     pred.clear();
  //     pred = get_cell_box(prob_data, i, j, side_, side_);
  //     // LOG(INFO) << "i: " << i << " j: " << j << " side: " << side_;
  //     // LOG(INFO) << "Box Pred: " << pred[0] << " " << pred[1] << " " << pred[2] << " " << pred[3];
  //     float best_iou = 0;
  //     for (int t = 0; t < 30; t++) {
  //       vector<Dtype> truth;
  //       truth.clear();
  //       Dtype x = label_data[t*5 + 1];
  //       Dtype y = label_data[t*5 + 2];
  //       Dtype w = label_data[t*5 + 3];
  //       Dtype h = label_data[t*5 + 4];

  //       if (!x) break;
  //       truth.push_back(x);
  //       truth.push_back(y);
  //       truth.push_back(w);
  //       truth.push_back(h);
  //       // LOG(INFO) << "Box Truth: " << truth[0] << " " << truth[1] << " " << truth[2] << " " << truth[3];
  //       Dtype iou = calc_iou(pred, truth);
  //       if(iou > best_iou) {
  //         best_iou = iou;
  //         // LOG(INFO) << "IOU: " << iou;
  //       }
  //     }
  //     if (best_iou > 0)
  //       fprintf(fp,"%d",1);
  //     else
  //       fprintf(fp,"%d",0);
  //     if (i != side_-1)
  //       fprintf(fp,",");
  //   }
  //   fprintf(fp,"\n");
  // }
  // fflush(fp);
  //*****************************************END DEBUG REGION LOSS************************************************************

  test_iter_++;
}

template <typename Dtype>
void ObjectLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if(propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* delta_data = delta_.cpu_data();
    caffe_copy(delta_.count(), delta_data, bottom_diff);
    // for (int b = 0; b < bottom[0]->num(); b++) {
    //   for (int n = 0; n < num_; n++) {
    //     int index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_,0);
    //     for (int i = index; i < index+(side_*side_*2); ++i) {
    //       bottom_diff[i] *= prob_data[i]*(1-prob_data[i]); //LOGISTIC GRADIENT
    //     }
    //     index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_, coords_);
    //     for (int i = index; i < index+(side_*side_); ++i) {
    //       bottom_diff[i] *= prob_data[i]*(1-prob_data[i]); //LOGISTIC GRADIENT
    //     }
    //   }
    // }

    for (int i=0; i<bottom[0]->count(); i++)
      bottom_diff[i]*= -prob_data[i]*(1-prob_data[i]); //LOGISTIC GRADIENT
    //************************DEBUG REGION BACKPROP**********************************
    // char filename[200];
    // sprintf(filename, "VerifyObject/object_delta_back_train_%d_%d_%d_%d_%d.csv", train_iter_, bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));
    // FILE *fp = fopen(filename, "w");
    // fp = fopen(filename, "w");
    // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
    // for (int i = 0; i < bottom[0]->shape(1)*bottom[0]->shape(2); i++){
    //   int spatialSize = bottom[0]->shape(3);
    //   int j = i*spatialSize;
    //   for(; j < ((i+1)*spatialSize)-1;j++){
    //     fprintf(fp,"%f,",bottom_diff[j]);
    //   }
    //   fprintf(fp,"%f\n",bottom_diff[j]);
    // }
    // fflush(fp);
    //**********************END DEBUG REGION BACKPROP********************************
    train_iter_++;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ObjectLossLayer);
#endif

INSTANTIATE_CLASS(ObjectLossLayer);
REGISTER_LAYER_CLASS(ObjectLoss);

}