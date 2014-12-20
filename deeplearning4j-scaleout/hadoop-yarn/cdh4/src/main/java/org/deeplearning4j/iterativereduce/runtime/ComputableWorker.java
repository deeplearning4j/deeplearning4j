package org.deeplearning4j.iterativereduce.runtime;

import org.deeplearning4j.iterativereduce.runtime.io.RecordParser;
import org.apache.hadoop.conf.Configuration;

import java.util.List;

public interface ComputableWorker<T extends Updateable> {
  void setup(Configuration c);
  T compute(List<T> records);
  T compute();
  // dont know a better way to do this currently
  void setRecordParser(RecordParser r);
  T getResults();
  void update(T t);
  

}
