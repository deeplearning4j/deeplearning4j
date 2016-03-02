package org.nd4j.jita.allocator.time;

/**
 * @author raver119@gmail.com
 */
public interface Ring {

    float getAverage();

    void store(float rate);

    void store(double rate);
}
