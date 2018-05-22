package org.nd4j.jita.allocator.time.providers;

import org.nd4j.jita.allocator.time.TimeProvider;

/**
 * @author raver119@gmail.com
 */
public class MillisecondsProvider implements TimeProvider {
    /**
     * This methods returns time in some, yet unknown, quants.
     *
     * @return
     */
    @Override
    public long getCurrentTime() {
        return System.currentTimeMillis();
    }
}
