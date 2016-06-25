package org.deeplearning4j.spark.time;


public interface TimeSource {

    /**
     * Get the current time in milliseconds, according to this TimeSource
     * @return Current time, since epoch
     */
    long getCurrentTimeMillis();

    //TODO add methods related to accuracy etc

}
