package org.nd4j.autodiff.listeners;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A base/abstract {@link Listener} with all methods implemented as no-op.
 * Extend this for custom listeners to selectively override only the required methods
 *
 * @author Alex Black
 */
public abstract class BaseListener implements Listener {

    @Override
    public void epochStart(SameDiff sd, At at) {
        //No op
    }

    @Override
    public void epochEnd(SameDiff sd, At at) {
        //No op
    }

    @Override
    public void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlMs) {
        //No op
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        //No op
    }

    @Override
    public void preOpExecution(SameDiff sd, At at, boolean training, SameDiffOp op) {
        //No op
    }

    @Override
    public void opExecution(SameDiff sd, At at, boolean training, SameDiffOp op, INDArray[] outputs) {
        //No op
    }

    @Override
    public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
        //No op
    }
}
