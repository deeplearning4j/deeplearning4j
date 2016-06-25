package org.deeplearning4j.spark.time;


public class TimeSourceProvider {

    public static final String DEFAULT_TIMESOURCE_CLASS_NAME = NTPTimeSource.class.getName();
    public static final String TIMESOURCE_CLASSNAME_PROPERTY = "org.deeplearning4j.spark.time.TimeSource";

    /**
     * Get a TimeSource
     * the default TimeSource instance (default: {@link NTPTimeSource}
     *
     * @return TimeSource
     */
    public TimeSource getInstance(){
        String className = System.getProperty(TIMESOURCE_CLASSNAME_PROPERTY, DEFAULT_TIMESOURCE_CLASS_NAME);

        return getInstance(className);
    }

    /**
     * Get a specific TimeSource by class name
     *
     * @param className
     * @return
     */
    public TimeSource getInstance(String className){

    }
}
