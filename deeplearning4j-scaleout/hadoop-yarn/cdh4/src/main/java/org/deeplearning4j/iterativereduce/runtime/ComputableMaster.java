package org.deeplearning4j.iterativereduce.runtime;

import org.apache.hadoop.conf.Configuration;
import org.deeplearning4j.scaleout.api.ir.Updateable;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

/**
 * Master computable
 * @param <T>
 */
public interface ComputableMaster<T extends Updateable> {
  void setup(Configuration c);
  void complete(DataOutputStream out) throws IOException;
  T compute(Collection<T> workerUpdates, Collection<T> masterUpdates);
  T getResults();
}