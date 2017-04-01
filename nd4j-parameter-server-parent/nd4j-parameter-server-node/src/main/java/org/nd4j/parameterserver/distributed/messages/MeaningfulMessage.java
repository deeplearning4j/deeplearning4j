package org.nd4j.parameterserver.distributed.messages;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface MeaningfulMessage extends VoidMessage {

    INDArray getPayload();
}
