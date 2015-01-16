package org.deeplearning4j.iterativereduce.runtime;

import org.apache.hadoop.conf.Configuration;
import org.deeplearning4j.scaleout.api.ir.Updateable;


public abstract class AbstractComputableMaster<T extends Updateable> implements ComputableMaster<T> {
  
  protected Configuration conf;
  
  public void setup(Configuration c) {
    conf = c;
  }
}