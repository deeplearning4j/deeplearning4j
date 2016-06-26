package org.deeplearning4j.spark.time;

/**
 * A time source is an abstraction of system time away from the local system clock.
 * Typically it is used in distributed computing settings, to allow for different time implementations, such as NTP
 * over the internet (via {@link NTPTimeSource}, local synchronization (LAN only - not implemented), or simply using the
 * standard clock on each machine (System.currentTimeMillis() via {@link SystemClockTimeSource}.
 *
 * @author Alex Black
 */
public interface TimeSource {

    /**
     * Get the current time in milliseconds, according to this TimeSource
     * @return Current time, since epoch
     */
    long currentTimeMillis();

    //TODO add methods related to accuracy etc

}
