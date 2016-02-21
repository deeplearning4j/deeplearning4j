package org.nd4j.linalg.heartbeat.reports;

/**
 * @author raver119@gmail.com
 */
public enum Event {
    TRAINING, // local training event
    EXPORT, // export of model
    AVERAGING, // spark-based training event
}
