#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/layers/region_loss_layer.hpp"
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
vector<Dtype> get_region_box(const Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h){
  vector<Dtype> b;
  b.clear();
  b.push_back((i + x[index + 0*w*h]) / w);
  b.push_back((j + x[index + 1*w*h]) / h);
  b.push_back(exp(x[index + 2*w*h]) * biases[2*n] / w);
  b.push_back(exp(x[index + 3*w*h]) * biases[2*n+1] / h);
  return b;
}

template <typename Dtype>
Dtype delta_region_box(vector<Dtype> truth, const Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, Dtype* delta, float scale){
  vector<Dtype> pred;
  pred.clear();
  pred = get_region_box(x, biases, n, index, i, j, w, h);
        
  float iou = calc_iou(pred, truth);
  //LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," << pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," << truth[3];
  float tx = truth[0] * w - i; //0.5
  float ty = truth[1] * h - j; //0.5
  float tw = log(truth[2] * w / biases[2*n]); //truth[2]=biases/w tw = 0
  float th = log(truth[3] * h / biases[2*n + 1]); //th = 0
  
  delta[index + 0*w*h] = scale * (tx - x[index + 0*w*h]);
  delta[index + 1*w*h] = scale * (ty - x[index + 1*w*h]);
  delta[index + 2*w*h] = scale * (tw - x[index + 2*w*h]);
  delta[index + 3*w*h] = scale * (th - x[index + 3*w*h]);

  // LOG(INFO) << "Index: " << index << " xIndex: " << index+0*w*h << " yIndex: " << index+1*w*h
  //         << " wIndex: " << index+2*w*h << " hIndex: " << index+3*w*h;
  // LOG(INFO) << "Tx: " << tx << " Ty: " << ty << " Tw: " << tw << " Th: " << th;
  // LOG(INFO) << "Scale: " << scale << " x: " << x[index+0*w*h] << " y: " << x[index+1*w*h]
  //         << " w: " << x[index+2*w*h] << " h: " << x[index+3*w*h];
  return iou;
}

template <typename Dtype>
void delta_region_class(const Dtype* input_data, Dtype* delta, int index, int class_label, int classes, float scale, int stride, float* avg_cat)
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
void RegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter region_param(this->layer_param_);
  region_param.set_type("Region");
  RegionParameter param = this->layer_param_.region_param();
  region_layer_ = LayerRegistry<Dtype>::CreateLayer(region_param);
  region_bottom_vec_.clear();
  region_bottom_vec_.push_back(bottom[0]);
  region_top_vec_.clear();
  region_top_vec_.push_back(&prob_);
  region_layer_->SetUp(region_bottom_vec_, region_top_vec_);

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
void RegionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  region_layer_->Reshape(region_bottom_vec_, region_top_vec_);
  delta_.ReshapeLike(prob_);
  // top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the region box coords and class prob values.
  region_layer_->Forward(region_bottom_vec_,region_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* delta_data = delta_.mutable_cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  caffe_set(delta_.count(), Dtype(0.0), delta_data);
  
  //************************************DEBUG REGION OUTPUT LAYER****************************************
  // // LOG(INFO) << label_data[0] << "," << label_data[1] << "," << label_data[2] << ","
  // //       << label_data[3] << "," << label_data[4];
  // // LOG(INFO) << prob_data[0] << "," << prob_data[1] << "," << prob_data[2];
  // // LOG(INFO) << "Delta Shape: " << delta_.shape(0) << ","
  // //         << delta_.shape(1) << "," << delta_.shape(2) << ","
  // //         << delta_.shape(3);

  char filename[200];
  if (this->phase_ == TEST)
    sprintf(filename, "VerifyShuffle/region_output_test_%d_%d_%d_%d_%d.csv", test_iter_, prob_.shape(0), prob_.shape(1), prob_.shape(2), prob_.shape(3));
  else
    sprintf(filename, "VerifyShuffle/region_output_train_%d_%d_%d_%d_%d.csv", train_iter_, prob_.shape(0), prob_.shape(1), prob_.shape(2), prob_.shape(3));
  FILE *fp = fopen(filename, "w");
  fp = fopen(filename, "w");
  if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  for (int i = 0; i < prob_.shape(1)*prob_.shape(2); i++){
    int spatialSize = prob_.shape(3);
    int j = i*spatialSize;
    for(; j < ((i+1)*spatialSize)-1;j++){
      fprintf(fp,"%f,",prob_data[j]);
    }
    fprintf(fp,"%f\n",prob_data[j]);
  }
  fflush(fp);
  //********************************END DEBUG REGION OUTPUT LAYER****************************************


  float avg_iou = 0;
  float recall = 0;
  float avg_cat = 0;
  float avg_obj = 0;
  float avg_anyobj = 0;
  int count = 0;
  int class_count = 0;
  for (int b = 0; b < bottom[0]->num(); b++) {
    for(int j = 0; j < side_; j++) {
      for(int i = 0; i < side_; i++) {
        for(int n = 0; n < num_; n++) {
          // int box_index = b*side_*side_*num_*(num_classes_+coords_+1) + 
          //                 n*side_*side_*(num_classes_+coords_+1) +
          //                 j*side_ + i;
          int box_index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_+j*side_+i, 0);
          // LOG(INFO) << box_index << ": " << prob_data[box_index];
          vector<Dtype> pred;
          pred.clear();
          pred = get_region_box(prob_data, biases_, n, box_index, i, j, side_, side_);
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
          int obj_index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_+j*side_+i, coords_);
          avg_anyobj += prob_data[obj_index];
          delta_data[obj_index] = noobject_scale_ * (0 - prob_data[obj_index]);
          // LOG(INFO) << "Obj_prob: " << prob_data[obj_index] << "\tObj_Index: " << obj_index << "\tDelta: " << delta_data[obj_index] << "\tThresh: " << thresh_;
          if (best_iou > thresh_) {
            delta_data[obj_index] = 0;
          }

          // Code only executed in darknet framework in early stages of training
          vector<Dtype> truth;
          truth.clear();
          truth.push_back((i + .5) / side_);
          truth.push_back((j + .5) / side_);
          truth.push_back((biases_[2 * n]) / side_); //anchor boxes
          truth.push_back((biases_[2 * n + 1]) / side_);
          delta_region_box(truth, prob_data, biases_, n, box_index, i, j, side_, side_, delta_data, .01);
        }
      }
    }
    for (int t = 0; t < 30; t++) {
      vector<Dtype> truth;
      truth.clear();
      int class_label = label_data[b*30*5 + t*5 + 0];
      Dtype x = label_data[b*30*5 + t*5 + 1];
      Dtype y = label_data[b*30*5 + t*5 + 2];
      Dtype w = label_data[b*30*5 + t*5 + 3];
      Dtype h = label_data[b*30*5 + t*5 + 4];
      if (!x) break;
      truth.push_back(x);
      truth.push_back(y);
      truth.push_back(w);
      truth.push_back(h);
      float best_iou = 0;
      int best_n = 0;
      int i = truth[0] * side_; //match which i,j
      int j = truth[1] * side_;

      vector<Dtype> truth_shift;
      truth_shift.clear();
      truth_shift.push_back(0);
      truth_shift.push_back(0);
      truth_shift.push_back(w);
      truth_shift.push_back(h);

      for (int n = 0; n < num_; ++ n){
        int box_index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_+j*side_+i, 0);
        vector<Dtype> pred;
        pred.clear();
        pred = get_region_box(prob_data, biases_, n, box_index, i, j, side_, side_);

        // The following two lines should be conditioned on bias_match, leaving out since we always have 1
        pred[2] = biases_[2 * n] / side_;
        pred[3] = biases_[2 * n + 1] / side_;
        pred[0] = 0;
        pred[1] = 0;
        float iou = calc_iou(pred, truth_shift);
        if (iou > best_iou) {
          best_iou = iou;
          best_n = n;
        }
        // LOG(INFO) << "Box Truth: " << truth[0] << " " << truth[1] << " " << truth[2] << " " << truth[3];
        // LOG(INFO) << "Box index: " << box_index;
        // LOG(INFO) << "Box Pred: " << pred[0] << " " << pred[1] << " " << pred[2] << " " << pred[3];
      }
      int box_index = entry_index(side_, num_classes_, num_, coords_, b, best_n*side_*side_+j*side_+i, 0);
      float iou = delta_region_box(truth, prob_data, biases_, best_n, box_index, i, j, side_, side_, delta_data, coord_scale_ *  (2 - w*h));
      // LOG(INFO) << "CoordScale: " << coord_scale_ << " W: " << w << " H: " << h;
      // LOG(INFO) << "Box index: " << box_index << "\tIOU: " << iou;
      if (iou > 0.5) recall += 1;
      avg_iou += iou;
      int obj_index = entry_index(side_, num_classes_, num_, coords_, b, best_n*side_*side_+j*side_+i, coords_);
      avg_obj += prob_data[obj_index];
      delta_data[obj_index] = object_scale_ * (1 - prob_data[obj_index]);
      // LOG(INFO) << "ObjIndex: " << obj_index << " ObjectScale: " << object_scale_ 
      //       << " out: " << prob_data[obj_index] << " delta: " << delta_data[obj_index];

      int class_index = entry_index(side_, num_classes_, num_, coords_, b, best_n*side_*side_+j*side_+i, coords_ + 1);
      delta_region_class(prob_data, delta_data, class_index, class_label, num_classes_, class_scale_, side_*side_, &avg_cat);
      ++count;
      ++class_count;
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
  //   sprintf(filename, "VerifyTrain/region_delta_test_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // else
  //   sprintf(filename, "VerifyTrain/region_delta_train_%d_%d_%d_%d_%d.csv", train_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
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
  //   sprintf(filename, "VerifyTrain/region_loss_test_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // else
  //   sprintf(filename, "VerifyTrain/region_loss_train_%d_%d_%d_%d_%d.csv", test_iter_, delta_.shape(0), delta_.shape(1), delta_.shape(2), delta_.shape(3));
  // fp = fopen(filename, "w");
  // if(!fp) LOG(ERROR) << "Couldn't open file: " << filename;
  // fprintf(fp,"%f,",loss);
  // fflush(fp);
  //*****************************************END DEBUG REGION LOSS************************************************************

  test_iter_++;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    for (int b = 0; b < bottom[0]->num(); b++) {
      for (int n = 0; n < num_; n++) {
        int index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_,0);
        for (int i = index; i < index+(side_*side_*2); ++i) {
          bottom_diff[i] *= prob_data[i]*(1-prob_data[i]); //LOGISTIC GRADIENT
        }
        index = entry_index(side_, num_classes_, num_, coords_, b, n*side_*side_, coords_);
        for (int i = index; i < index+(side_*side_); ++i) {
          bottom_diff[i] *= prob_data[i]*(1-prob_data[i]); //LOGISTIC GRADIENT
        }
      }
    }
    for (int i = 0; i < bottom[0]->count(); i++)
      bottom_diff[i] *= -1;
    //************************DEBUG REGION BACKPROP**********************************
    // char filename[200];
    // sprintf(filename, "VerifyTrain/region_delta_back_train_%d_%d_%d_%d_%d.csv", train_iter_, bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));
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
STUB_GPU(RegionLossLayer);
#endif

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}