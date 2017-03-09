package org.deeplearning4j.spark.time;

import java.lang.reflect.Method;

/**
 * TimeSourceProvider: used to get a TimeSource via a static method.<br>
 * Defaults to the Network Time Protocol implementation {@link NTPTimeSource}, but can be switched to other implementations
 * via the {@link TimeSourceProvider#TIMESOURCE_CLASSNAME_PROPERTY} system property.
 *
 * @author Alex Black
 */
public class TimeSourceProvider {

    /**
     * Default class to use when getting a TimeSource instance
     */
    public static final String DEFAULT_TIMESOURCE_CLASS_NAME = NTPTimeSource.class.getName();

    /**
     * Name of the system property to set if the TimeSource type/class is to be customized
     */
    public static final String TIMESOURCE_CLASSNAME_PROPERTY = "org.deeplearning4j.spark.time.TimeSource";

    private TimeSourceProvider() {}

    /**
     * Get a TimeSource
     * the default TimeSource instance (default: {@link NTPTimeSource}
     *
     * @return TimeSource
     */
    public static TimeSource getInstance() {
        String className = System.getProperty(TIMESOURCE_CLASSNAME_PROPERTY, DEFAULT_TIMESOURCE_CLASS_NAME);

        return getInstance(className);
    }

    /**
     * Get a specific TimeSource by class name
     *
     * @param className Class name of the TimeSource to return the instance for
     * @return TimeSource instance
     */
    public static TimeSource getInstance(String className) {
        try {
            Class<?> c = Class.forName(className);
            Method m = c.getMethod("getInstance");
            return (TimeSource) m.invoke(null);
        } catch (Exception e) {
            throw new RuntimeException("Error getting TimeSource instance for class \"" + className + "\"", e);
        }
    }
}
