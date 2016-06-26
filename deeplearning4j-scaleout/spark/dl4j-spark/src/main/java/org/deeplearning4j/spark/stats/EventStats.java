package org.deeplearning4j.spark.stats;

import java.io.Serializable;

/**
 * Created by Alex on 26/06/2016.
 */
public interface EventStats extends Serializable {

    String getMachineID();

    String getJvmID();

    long getThreadID();

    long getStartTime();

    long getDurationMs();

}
