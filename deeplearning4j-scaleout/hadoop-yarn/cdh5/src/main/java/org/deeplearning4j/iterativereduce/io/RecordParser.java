package org.deeplearning4j.iterativereduce.io;


import org.deeplearning4j.iterativereduce.Updateable;

public interface RecordParser<T extends Updateable> {
  void reset();
  void parse();
  void setFile(String file, long offset, long length);
  void setFile(String file);
  boolean hasMoreRecords();
  T nextRecord();
  int getCurrentRecordsProcessed();
}