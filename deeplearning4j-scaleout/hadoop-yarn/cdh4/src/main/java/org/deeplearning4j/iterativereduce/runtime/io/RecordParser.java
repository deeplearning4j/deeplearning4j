package org.deeplearning4j.iterativereduce.runtime.io;


import org.deeplearning4j.iterativereduce.runtime.Updateable;

public interface RecordParser<T extends Updateable> {
  void reset();
  void parse();
  void setFile(String file, long offset, long length);
  void setFile(String file);
  boolean hasMoreRecords();
  T nextRecord();
  int getCurrentRecordsProcessed();
}