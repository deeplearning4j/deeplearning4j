package org.nd4j.jita.allocator.time;

/**
 * @author raver119@gmail.com
 */
public interface DecayingTimer {

    /**
     * This method notifies timer about event
     */
    void triggerEvent();

    /**
     * This method returns average frequency of events happened within predefined timeframe
     * @return
     */
    double getFrequencyOfEvents();

    /**
     * This method returns total number of events happened withing predefined timeframe
     * @return
     */
    long getNumberOfEvents();
}
