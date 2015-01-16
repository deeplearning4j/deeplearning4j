package org.deeplearning4j.iterativereduce.runtime;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.RecordReader;
import org.deeplearning4j.scaleout.api.ir.Updateable;

import java.util.List;

public interface ComputableWorker<T extends Updateable> {
  void setup(Configuration c);
  T compute(List<T> records);
  T compute();
  void setRecordReader(RecordReader r);
  T getResults();
  void update(T t);
  

}
