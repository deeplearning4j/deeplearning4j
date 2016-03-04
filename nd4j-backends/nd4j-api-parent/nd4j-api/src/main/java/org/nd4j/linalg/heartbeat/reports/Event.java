package org.nd4j.linalg.heartbeat.reports;

/**
 * @author raver119@gmail.com
 */
public enum Event {
    STANDALONE, // local training event
    EXPORT, // export of model
    SPARK, // spark-based training event
}
