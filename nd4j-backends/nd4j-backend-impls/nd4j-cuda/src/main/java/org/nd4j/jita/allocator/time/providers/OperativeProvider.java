package org.nd4j.jita.allocator.time.providers;

import org.nd4j.jita.allocator.time.TimeProvider;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class OperativeProvider implements TimeProvider {
    private AtomicLong time = new AtomicLong(0);


    /**
     * This methods returns time in some, yet unknown, quants.
     *
     * @return
     */
    @Override
    public long getCurrentTime() {
        return time.incrementAndGet();
    }
}
