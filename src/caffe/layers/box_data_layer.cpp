#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
BoxDataLayer<Dtype>::BoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
BoxDataLayer<Dtype>::~BoxDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  LOG(INFO) << "TOP_SHAPE " << top_shape[0] << "," << top_shape[1] << ","
          << top_shape[2] << "," << top_shape[3];
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    LOG(INFO) << "LABELS " << this->prefetch_.size();
    vector<int> label_shape(1, batch_size);
    int num_objects = 30; // max number of objects in a class
    int class_coords = 5; // (class + coords, 1+4)
    label_shape.push_back(num_objects);
    label_shape.push_back(class_coords);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}

template <typename Dtype>
bool BoxDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void BoxDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void BoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  Datum datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<float> box_labels;
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {

      Dtype* top_label = batch->label_.mutable_cpu_data();
      top_label[item_id] = datum.label();
      // LOG(INFO) << "KEY: " << cursor_->key();
      int index = (datum.float_data_size()/150)-1;
      for (int i = 0; i < 150; ++i) {
        top_label[i] = datum.float_data((index*150)+i);
      }
      // LOG(INFO) << "\tFLOAT DATA Size: " << (datum.float_data_size()/150)-1;
      // LOG(INFO) << "\tTOP_LABEL: " << batch->label_.count();
      // LOG(INFO) << "\t\tLABEL VALUE: " << datum.float_data(0) << "," 
      //     << datum.float_data(1) << "," << datum.float_data(2) << ","
      //     << datum.float_data(3) << "," << datum.float_data(4);
      // LOG(INFO) << "\t\tLABEL VALUE: " << datum.float_data(5) << "," 
      //     << datum.float_data(6) << "," << datum.float_data(7) << ","
      //     << datum.float_data(8) << "," << datum.float_data(9);
      // LOG(INFO) << "\t\tLABEL VALUE: " << datum.float_data(10) << "," 
      //     << datum.float_data(11) << "," << datum.float_data(12) << ","
      //     << datum.float_data(13) << "," << datum.float_data(14);
      // LOG(INFO) << "\t\tLABEL VALUE: " << datum.float_data(15) << "," 
      //     << datum.float_data(16) << "," << datum.float_data(17) << ","
      //     << datum.float_data(18) << "," << datum.float_data(19);
      // LOG(INFO) << "\t\tLABEL VALUE: " << datum.float_data(20) << "," 
      //     << datum.float_data(21) << "," << datum.float_data(22) << ","
      //     << datum.float_data(23) << "," << datum.float_data(24);
      // LOG(INFO) << "\t\t\tOFFSET: " << this->transformed_data_.data_at(0,0,144,144);
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(BoxDataLayer);
REGISTER_LAYER_CLASS(BoxData);

}  // namespace caffe
