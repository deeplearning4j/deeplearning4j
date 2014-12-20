package org.deeplearning4j.iterativereduce;

import java.nio.ByteBuffer;

public interface Updateable<T> {
  ByteBuffer toBytes();
  void fromBytes(ByteBuffer b);
  void fromString(String s);
  T get();
  void set(T t);
//  void setIterationState(int IterationNumber, int BatchNumber);
//  int getGlobalIterationNumber();
//  int getGlobalBatchNumber();
}