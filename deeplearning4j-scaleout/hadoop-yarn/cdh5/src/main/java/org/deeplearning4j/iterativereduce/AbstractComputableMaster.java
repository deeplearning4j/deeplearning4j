package org.deeplearning4j.iterativereduce;

import org.apache.hadoop.conf.Configuration;


public abstract class AbstractComputableMaster<T extends Updateable> implements ComputableMaster<T> {
  
  private Configuration conf;
  
  public void setup(Configuration c) {
    conf = c;
  }
}