package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.compression.NDArrayCompressor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * This MessageHandler implementation is suited for debugging mostly, but still can be used in production environment if you really want that.
 * Basic idea: updates are encoded before sharing.
 *
 * This handler is used as basement for distributed handler though.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class EncodingHandler implements MessageHandler {
    protected transient GradientsAccumulator accumulator;
    protected double threshold;
    protected NDArrayCompressor compressor;

    public EncodingHandler() {
        this(1e-3);
    }

    public EncodingHandler(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public void initialize(@NonNull GradientsAccumulator accumulator) {
        this.accumulator = accumulator;

        compressor = Nd4j.getCompressor().getCompressor("THRESHOLD");
        if (compressor == null)
            throw new ND4JIllegalStateException("Can't find Threshold compressor implementation!");

        compressor.configure(threshold);
    }

    public INDArray encodeUpdates(INDArray updates) {
        // special op should be called here for encoding

        return compressor.compress(updates);
    }

    public INDArray decodeUpdates(INDArray message) {
        // special op should be called here for decoding

        return compressor.decompress(message);
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
    public boolean broadcastUpdates(INDArray updates) {
        /*
            we want to do 2 things here:
            1) encode updates
            2) send them somewhere
         */
        INDArray message = encodeUpdates(updates);
        if (message != null) {
            sendMessage(message);
            return true;
        } else return false;
    }
}
