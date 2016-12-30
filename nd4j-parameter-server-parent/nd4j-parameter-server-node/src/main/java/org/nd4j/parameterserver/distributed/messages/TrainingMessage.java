package org.nd4j.parameterserver.distributed.messages;

/**
 * @author raver119@gmail.com
 */
public interface TrainingMessage extends VoidMessage {
    // dummy interface
    byte getCounter();
}
