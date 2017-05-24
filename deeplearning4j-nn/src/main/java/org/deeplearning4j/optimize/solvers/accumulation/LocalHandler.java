package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * MessageHandler implementation suited for ParallelWrapper running on single box
 *
 * PLEASE NOTE: This handler does NOT provide any network connectivity.
 *
 * @author raver119@gmail.com
 */
public class LocalHandler implements MessageHandler {
    protected transient GradientsAccumulator accumulator;

    public LocalHandler() {
        //
    }

    @Override
    public void initialize(@NonNull GradientsAccumulator accumulator) {
        this.accumulator = accumulator;
    }

    @Override
    public void broadcastUpdates(INDArray updates) {
        // we just loop back data immediately
        accumulator.receiveUpdate(updates);
    }
}
