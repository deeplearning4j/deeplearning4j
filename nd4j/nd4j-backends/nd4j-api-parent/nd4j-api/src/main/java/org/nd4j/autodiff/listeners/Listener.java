package org.nd4j.autodiff.listeners;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A {@link SameDiff} listener interface that is called during every iteration of training or inference
 *
 * @author Alex Black
 * @see BaseListener BaseListener, for extending
 */
public interface Listener {

    /**
     * Called at the start of every epoch, when fitting from an iterator
     *
     * @param sd The SameDiff instance
     * @param at Current iteration/epoch etc
     */
    void epochStart(SameDiff sd, At at);

    /**
     * Called at the end of every epoch, when fitting from an iterator
     *
     * @param sd The SameDiff instance
     * @param at Current iteration/epoch etc
     */
    void epochEnd(SameDiff sd, At at);

    /**
     * Called at the start of every iteration (minibatch), before any operations have been executed
     *
     * @param sd The SameDiff instance
     * @param at Current iteration/epoch etc
     */
    void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlTimeMs);

    /**
     * Called at the end of every iteration, after all operations (including updating parameters) has been completed
     *
     * @param sd      The SameDiff instance
     * @param at      Current iteration/epoch etc
     * @param dataSet The current dataset (minibatch) used for training
     * @param loss    The loss value for the current minibatch
     */
    void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss);

    /**
     * Called just before each operation is executed (native code called, etc) - after all inputs etc have been set
     *
     * @param sd      The SameDiff instance
     * @param at      Current iteration/epoch etc
     * @param op      Operation that has just been executed
     */
    void preOpExecution(SameDiff sd, At at, boolean training, SameDiffOp op);

    /**
     * Called at the end of each operation execution
     *
     * @param sd      The SameDiff instance
     * @param at      Current iteration/epoch etc
     * @param op      Operation that has just been executed
     * @param outputs The output arrays for the just-executed operation
     */
    void opExecution(SameDiff sd, At at, boolean training, SameDiffOp op, INDArray[] outputs);

    /**
     * Called just before each parameter is to be updated - i.e., just before each parameter is modified
     *
     * @param sd     SameDiff instance
     * @param at     The current iteration/epoch etc
     * @param v      Variable about to be updated during backprop
     * @param update The array representing the update (i.e., the gradient after applying learning rate, momentum, etc)
     */
    void preUpdate(SameDiff sd, At at, Variable v, INDArray update);

}
