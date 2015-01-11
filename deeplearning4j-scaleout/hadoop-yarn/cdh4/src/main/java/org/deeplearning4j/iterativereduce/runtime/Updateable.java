package org.deeplearning4j.iterativereduce.runtime;


import java.nio.ByteBuffer;

/**
 * An updateable object
 * @param <T> the type
 */
public interface Updateable<T> {
  ByteBuffer toBytes();
  void fromBytes(ByteBuffer b);
  void fromString(String s);
  T get();
  void set(T t);

}