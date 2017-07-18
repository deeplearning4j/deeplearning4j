package org.deeplearning4j.optimize.solvers.accumulation;

/**
 * @author raver119@gmail.com
 */
public interface Registerable {

    /**
     * This method notifies producer about number of consumers for the current consumption cycle
     * @param numConsumers
     */
    void registerConsumers(int numConsumers);

    /**
     * This method enables/disables bypass mode
     *
     * @param reallyFallback
     */
    void fallbackToSingleConsumerMode(boolean reallyFallback);
}
