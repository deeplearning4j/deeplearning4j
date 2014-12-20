package org.deeplearning4j.iterativereduce;

import org.apache.hadoop.conf.Configuration;


public abstract class AbstractComputableWorker<T extends Updateable> implements
    ComputableWorker<T> {
  
  private Configuration conf;
  
  public void setup(Configuration c) {
    conf = c;
  }
}