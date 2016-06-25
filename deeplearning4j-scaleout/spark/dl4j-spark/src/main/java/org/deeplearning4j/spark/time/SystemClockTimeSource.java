package org.deeplearning4j.spark.time;


/**
 * Created by Alex on 25/06/2016.
 */
public class SystemClockTimeSource implements TimeSource {
    public long getCurrentTimeMillis() {
        return System.currentTimeMillis();
    }
}
