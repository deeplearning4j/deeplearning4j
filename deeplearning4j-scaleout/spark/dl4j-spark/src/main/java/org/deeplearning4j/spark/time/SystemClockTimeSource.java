package org.deeplearning4j.spark.time;


/**
 * A {@link TimeSource} implementation that is identical to calling {@link System#currentTimeMillis()}
 *
 * @author Alex Black
 */
public class SystemClockTimeSource implements TimeSource {
    public long getCurrentTimeMillis() {
        return System.currentTimeMillis();
    }
}
