package org.nd4j.parameterserver.distributed.logic;

/**
 * @author raver119@gmail.com
 */
public interface SequenceProvider {
    Long getNextValue();
}
