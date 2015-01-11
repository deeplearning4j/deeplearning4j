package org.deeplearning4j.iterativereduce.runtime;

import org.apache.hadoop.conf.Configuration;


public abstract class AbstractComputableMaster<T extends Updateable> implements ComputableMaster<T> {
  
  protected Configuration conf;
  
  public void setup(Configuration c) {
    conf = c;
  }
}