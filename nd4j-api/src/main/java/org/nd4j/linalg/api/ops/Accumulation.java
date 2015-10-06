/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;

import java.util.List;

/**
 * An accumulation is an op that given:<br>
 * x -> the origin ndarray<br>
 * y -> the pairwise ndarray<br>
 * n -> the number of times to accumulate<br>
 * <p/>
 * <p/>
 * Of note here in the extra arguments.
 * <p/>
 * An accumulation (or reduction in some terminology)
 * has a concept of a starting value.
 * <p/>
 * The starting value is the initialization of the solution
 * to the operation.
 * <p/>
 * An accumulation should always have the extraArgs()
 * contain the zero value as the first value.
 * <p/>
 * This allows the architecture to generalize to different backends
 * and gives the implementer of a backend a way of hooking in to
 * passing parameters to different engines.<br>
 *
 * Note that Accumulation op implementations should be stateless
 * (other than the final result and x/y/z/n arguments) and hence threadsafe,
 * such that they may be parallelized using the update, combineSubResults and
 * set/getFinalResults methods.
 * @author Adam Gibson
 */
public interface Accumulation extends Op {

    /** Do one accumulation update for a single-argument accumulation, given the
     * current accumulation value and another value to be processed/accumulated
     * @param accum The current accumulation value
     * @param x The next/new value to be processed/accumulated
     * @return The updated accumulation value
     */
    double update(double accum, double x);

    /** Do an accumulation update for a pair-wise (op(x,y)) accumulation, given the
     * current accumulation value and a pair of values to be processed/accumulated
     * @param accum The current accumulation value
     * @param x The next/new x value to be processed/accumulated
     * @param y The next/new y value to be processed/accumulated
     * @return the updated accumulation value
     */
    double update(double accum, double x, double y);

    /**@see #update(double, double) */
    float update(float accum, float x);

    /**@see #update(double, double, double)*/
    float update(float accum, float x, float y);

    /** Complex update.
     * @see #update(double, double)
     */
    IComplexNumber update( IComplexNumber accum, double x);

    /** Complex update.
     * @see #update(double, double, double)
     */
    IComplexNumber update( IComplexNumber accum, double x, double y);

    /** Complex update.
     * @see #update(double, double)
     */
    IComplexNumber update( IComplexNumber accum, IComplexNumber x);

    /** Complex update.
     * @see #update(double, double, double)
     */
    IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y);

    /** Complex update.
     * @see #update(double, double, double)
     */
    IComplexNumber update( IComplexNumber accum, IComplexNumber x, double y);

    /** Combine sub-results, when the Accumulation operation is split and executed in
     * parallel.<br>
     * Sometimes this is the same as the update operation (i.e., in sum operation), but often
     * it is not (for example, in L2 norm, sqrt(1/n * sum_i x_i^2).<br>
     * @param first The first sub-value to be combined
     * @param second The second sub-value to be combined
     * @return The combined result
     */
    double combineSubResults(double first, double second);

    /**@see #combineSubResults(double, double) */
    float combineSubResults(float first, float second);

    /**@see #combineSubResults(double, double) */
    IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second);

    /** Get and set the final result.<br>
     * The final result is also stored in the Accumulation op itself (and can be retrieved
     * by getFinalResult())
     * In some Accumulation operations, a final operation must be done on the accumulated
     * result. For example, division by n in a mean operation: the accumulated result is
     * the sum, whereas the final result is the sum/n.
     * @param accum The accumulated result
     * @return The final result
     */
    double getAndSetFinalResult(double accum);

    /** @see #getAndSetFinalResult(double) */
    float getAndSetFinalResult(float accum);

    /** Complex version of getAndSetFinalResult().
     * @see #getAndSetFinalResult(double)
     */
    IComplexNumber getAndSetFinalResult(IComplexNumber accum);

    /** Get the final result (may return null if getAndSetFinalResult has not
     * been called, or for accumulation ops on complex arrays)
     */
    Number getFinalResult();

    /** Essentially identical to {@link #getFinalResult()}, maintained to avoid breaking
     * implementations created before Accumulation ops were refactored.
     * May be removed in a future release.
     */
    @Deprecated
    Number currentResult();

    IComplexNumber getFinalResultComplex();

    void setFinalResult(Number number);

    void setFinalResultComplex(IComplexNumber number);


    /**Initial value (used to initialize the accumulation op)
     * @return the initial value
     */
    double zeroDouble();

    /** Initial value (used to initialize the accumulation op) */
    float zeroFloat();

    /**Complex initial value
     *@return the complex initial value
     */
    IComplexNumber zeroComplex();

    /**Get an AccumulationDataBufferTask. Used (internally) for parallelizing Accumulation operations.
     * Note that most Accumulation implementations can simply inherit BaseAccumulation.getAccumulationOpDataBufferTask,
     * however this can be overridden in certain cases for performance reasons
     *
     * @param threshold The parallelism threshold for breaking an AccumulationDataBufferTask into smaller sub-tasks
     *                  for parallel execution
     * @param n The number of operations / length
     * @param x x DataBuffer
     * @param y y DataBuffer (may be null for accumulations with a single argument)
     * @param offsetX Offset of the first element to be processed in the X databuffer
     * @param offsetY Offset of the first element to be processed in the Y databuffer (not used if y is null)
     * @param incrX Increment between elements in the X databuffer
     * @param incrY Increment between elements in the Y databuffer (not used if y is null)
     * @param outerTask Whether this DataBufferTask should be considered an 'outer' task or an 'inner' task
     *                  Generally, if the DataBufferTask is created by another DataBufferTask, outerTask==false.
     *                  An 'outer' task could be called a root/primary task.
     *                  In practice, generally if outerTask==true, getAndSetFinalResult is called and the accumulation
     *                  returned is 'final' in the sense of being divided by n, or sqrt(accum) etc if such a final
     *                  operation is required
     * @return The AccumulationDataBufferTask
     */
    AccumulationDataBufferTask getAccumulationOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y,
                                                               int offsetX, int offsetY, int incrX, int incrY, boolean outerTask);

    /**Get an AccumulationDataBufferTask. Used (internally) for parallelizing Accumulation operations.
     * Unlike the other getAccumulationOpDataBufferTask, the resulting AccumulationDataBufferTask should
     * calculate a 1d tensor first.
     * Note that most Accumulation implementations can simply inherit BaseAccumulation.getAccumulationOpDataBufferTask,
     * however this can be overridden in certain cases for performance reasons
     *
     * @param tensorNum The number/index of the 1d tensor that should be calculated (i.e., x.tensorAlongDimension(tensorNum,tensorDim))
     * @param tensorDim The dimension that the 1d tensor should be calculated along
     * @param parallelThreshold The threshold for parallelism (for breaking an AccumulationDataBufferTask into smaller
     *                          parallel ops)
     * @param x x INDArray
     * @param y y INDArray (may be null)
     * @param outerTask Whether this DataBufferTask should be considered an 'outer' task or 'inner task.
     *                  Generally, if the DataBufferTask is created by another DataBufferTask, outerTask==false.
     *                  An 'outer' task could be called a root/primary task.
     *                  In practice, generally if outerTask==true, getAndSetFinalResult is called and the accumulation
     *                  returned is 'final' in the sense of being divided by n, or sqrt(accum) etc if such a final
     *                  operation is required
     * @return The AccumulationDataBufferTask
     * @see #getAccumulationOpDataBufferTask(int, int, DataBuffer, DataBuffer, int, int, int, int, boolean)
     */
    AccumulationDataBufferTask getAccumulationOpDataBufferTask(int tensorNum, int tensorDim, int parallelThreshold,
                                                               INDArray x, INDArray y, boolean outerTask);
}
