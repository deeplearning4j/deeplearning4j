/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.listeners;

import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A {@link SameDiff} listener interface that is called during every iteration of training or inference
 *
 * @author Alex Black
 * @see BaseListener BaseListener, for extending only the required methods (all others are no-op)
 * @see BaseEvaluationListener BaseEvaluationListener, for extending if you want to use evaluations
 */
public interface Listener {


    /**
     * Required variables for this listener.
     * <p>
     * Used to ensure these variables end up in the minimum required subgraph calculated by {@link org.nd4j.autodiff.samediff.internal.InferenceSession}.
     * Otherwise, if the variables weren't required by a loss variable, they would not be calculated.
     * <p>
     * Any variables in here are guaranteed to have {@link Listener#activationAvailable(SameDiff, At, MultiDataSet, SameDiffOp, String, INDArray)}
     * called for them, regardless of whether they would normally be calculated or not.
     */
    ListenerVariables requiredVariables(SameDiff sd);

    /**
     * Returns whether this listener is active during the given operation. If this returns false for the given operation,
     * those listener methods will not be called.
     */
    boolean isActive(Operation operation);

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
     * @param sd              The SameDiff instance
     * @param at              Current iteration/epoch etc
     * @param lossCurve       The losses so far
     * @param epochTimeMillis How long this epoch took
     * @return ListenerResponse.STOP to stop training, CONTINUE or null to continue
     */
    ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis);

    /**
     * Called after the end of every epoch, once validation evaluation is done, when training
     *
     * @param sd                   The SameDiff instance
     * @param at                   Current iteration/epoch etc
     * @param validationTimeMillis How long validation took for this epoch
     * @return ListenerResponse.STOP to stop training, CONTINUE or null to continue
     */
    ListenerResponse validationDone(SameDiff sd, At at, long validationTimeMillis);

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
     * @param loss    The loss value for the current minibatch.  Will be null except for during training
     */
    void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss);

    /**
     * Called at the start of an operation, e.g. training or validation
     *
     * @param sd The SameDiff instance
     * @param op The operation being started
     */
    void operationStart(SameDiff sd, Operation op);

    /**
     * Called at the end of an operation, e.g. training or validation
     *
     * @param sd The SameDiff instance
     * @param op The operation being started
     */
    void operationEnd(SameDiff sd, Operation op);

    /**
     * Called just before each operation is executed (native code called, etc) - after all inputs etc have been set
     *
     * @param sd The SameDiff instance
     * @param at Current iteration/epoch etc
     * @param op Operation that has just been executed
     */
    void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext);

    /**
     * Called at the end of each operation execution<br>
     * <p>
     * Note: Outputs will most likely be freed later, use detach() if you need to save it.
     *
     * @param sd      The SameDiff instance
     * @param at      Current iteration/epoch etc
     * @param batch   The batch's input data.  May be null if not called with a batch
     * @param op      Operation that has just been executed
     * @param outputs The output arrays for the just-executed operation
     */
    void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs);

    /**
     * Called when any activation becomes available.
     * <p>
     * The activation will most likely be freed later, use dup() if you need to save it.<br>
     * <br>
     * Note that this method will be called when any activation becomes available, not just ones from {@link #requiredVariables(SameDiff)}<br>
     * It is guaranteed to be called for variables from requiredVariables().<br>
     * <br>
     * Note that the activations here overlap with {@link #opExecution(SameDiff, At, MultiDataSet, SameDiffOp, OpContext, INDArray[])} -
     * both contain the same information/arrays
     *
     * @param sd         The SameDiff instance
     * @param at         Current iteration/epoch etc
     * @param batch      The batch's input data.  May be null if not called with a batch
     * @param op         Operation that has just been executed
     * @param varName    The name of the variable
     * @param activation The variable's activation
     */
    void activationAvailable(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, String varName, INDArray activation);

    /**
     * Called just before each parameter is to be updated - i.e., just before each parameter is modified.
     *
     * @param sd     SameDiff instance
     * @param at     The current iteration/epoch etc
     * @param v      Variable about to be updated during backprop
     * @param update The array representing the update (i.e., the gradient after applying learning rate, momentum, etc)
     */
    void preUpdate(SameDiff sd, At at, Variable v, INDArray update);

}
