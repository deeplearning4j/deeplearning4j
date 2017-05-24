package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This MessageHandler implementation is suited for debugging mostly, but still can be used in production environment if you really want that.
 * Basic idea: updates are encoded before sharing.
 *
 * This handler is used as basement for distributed handler though.
 *
 * @author raver119@gmail.com
 */
public class EncodingHandler implements MessageHandler {
    protected transient GradientsAccumulator accumulator;
    protected double threshold;

    public EncodingHandler() {
        this(1e-3);
    }

    public EncodingHandler(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public void initialize(@NonNull GradientsAccumulator accumulator) {
        this.accumulator = accumulator;
    }

    public INDArray encodeUpdates(INDArray updates) {
        // special op should be called here for encoding
        return null;
    }

    public INDArray decodeUpdates(INDArray message) {
        // special op should be called here for decoding
        return null;
    }

    /**
     * This method does loops encoded data back to updates queue
     * @param message
     */
    protected void sendMessage(INDArray message) {
        INDArray update = decodeUpdates(message);
        accumulator.receiveUpdate(update);
    }

    @Override
    public void broadcastUpdates(INDArray updates) {
        /*
            we want to do 2 things here:
            1) encode updates
            2) send them somewhere
         */
        INDArray message = encodeUpdates(updates);
        sendMessage(message);
    }
}
