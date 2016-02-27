package org.nd4j.jita.allocator.time;

/**
 * @author raver119@gmail.com
 */
public interface TimeProvider {

    /**
     * This methods returns time in some, yet unknown, quants.
     *
     *
     * @return
     */
    long getCurrentTime();
}
