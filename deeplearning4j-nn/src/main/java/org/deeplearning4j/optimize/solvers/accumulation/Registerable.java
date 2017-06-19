package org.deeplearning4j.optimize.solvers.accumulation;

/**
 * @author raver119@gmail.com
 */
public interface Registerable {

    void register(int numConsumers);
}
