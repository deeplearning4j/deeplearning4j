package org.deeplearning4j.iterativereduce.runtime;

import org.apache.hadoop.conf.Configuration;
import org.deeplearning4j.scaleout.api.ir.Updateable;


public abstract class AbstractComputableWorker<T extends Updateable> implements
    ComputableWorker<T> {
  
  private Configuration conf;
  
  public void setup(Configuration c) {
    conf = c;
  }
}