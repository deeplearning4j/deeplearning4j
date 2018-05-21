package org.nd4j.parameterserver.distributed.messages;

/**
 * This interface describes number of actions happening within VoidParameterServer in distributed manner
 *
 * @author raver119@gmail.com
 */
public interface Chain {

    long getOriginatorId();

    long getTaskId();

    void addElement(VoidMessage message);
}
