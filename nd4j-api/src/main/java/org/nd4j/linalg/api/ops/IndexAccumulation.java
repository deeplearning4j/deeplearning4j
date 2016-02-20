package org.nd4j.linalg.api.ops;

import org.apache.commons.math3.util.Pair;

/**An index accumulation is an operation that returns an index within
 * a NDArray.<br>
 * Examples of IndexAccumulation operations include finding the index
 * of the minimim value, index of the maximum value, index of the first
 * element equal to value y, index of the maximum pair-wise difference
 * between two NDArrays X and Y etc.<br>
 *
 * Index accumulation is similar to {@link Accumulation} in that both are
 * accumulation/reduction operations, however index accumulation returns
 * an integer corresponding to an index, rather than a real (or complex)
 * value.<br>
 *
 * Index accumulation operations generally have 3 inputs:<br>
 * x -> the origin ndarray<br>
 * y -> the pairwise ndarray (frequently null/not applicable)<br>
 * n -> the number of times to accumulate<br>
 *
 * Note that IndexAccumulation op implementations should be stateless
 * (other than the final result and x/y/n arguments) and hence threadsafe,
 * such that they may be parallelized using the update, combineSubResults
 * and set/getFinalResults methods.
 */
public interface IndexAccumulation extends Op {

    /**Do one accumulation update. Given the current accumulation value (for
     * example, maximum value found so far in an 'index of maximum value'
     * operation) and another value and its index (x/xIdx), update the
     * accumulation index.<br>
     * If the accumulation value is required in addition to its index,
     * use the Op.op(...) methods.
     * @param accum The value at the accumulation index
     * @param accumIdx The accumulation index
     * @param x A value to be processed/accumulated
     * @param xIdx The index of the value to be processed/accumulated
     * @return The new accumulation index
     */
    int update(double accum, int accumIdx, double x, int xIdx);

    /**@see #update(double, int, double, int)
     */
    int update(float accum, int accumIdx, float x, int xIdx);

    /** Do one accumulation update for a pair-wise (2 input) index accumulation
     * @see #update(double, int, double, int)
     */
    int update(double accum, int accumIdx, double x, double y, int idx);

    /** @see #update(double, int, double, double, int) */
    int update(float accum, int accumIdx, float x, float y, int idx);

    /** Accumulation update for complex numbers.
     * @see #update(double, int, double, int)
     */
    int update(IComplexNumber accum, int accumIdx, IComplexNumber x, int xIdx);

    /** Accumulation update for complex numbers.
     * @see #update(double, int, double, int)
     */
    int update(IComplexNumber accum, int accumIdx, double x, int idx);

    /** Accumulation update for pairswise (2 input) complex numbers.
     * @see #update(double, int, double, int)
     */
    int update(IComplexNumber accum, int accumIdx, double x, double y, int idx);

    /** Accumulation update for pairswise (2 input) complex numbers.
     * @see #update(double, int, double, int)
     */
    int update(IComplexNumber accum, int accumIdx, IComplexNumber x, IComplexNumber y, int idx);

    /** Combine sub-results, when the index accumulation is split and executed in parallel.
     * Often this is the same as the update operations (i.e., when finding index of maximum
     * value for example), but sometimes not. <br>
     * Generally used internally by OpExecutors etc.
     * @param first Value of one sub-accumulation (i.e., partial from one thread)
     * @param idxFirst Index of the first value
     * @param second Value of another sub-accumulation (i.e., partial accumulation from another thread)
     * @param idxSecond Index of the second value
     * @return Combined/accumulated index
     */
    int combineSubResults(double first, int idxFirst, double second, int idxSecond);

    /** @see #combineSubResults(double, int, double, int) */
    int combineSubResults(float first, int idxFirst, float second, int idxSecond);

    /** @see #combineSubResults(double, int, double, int) */
    Pair<Double,Integer> combineSubResults(Pair<Double,Integer> first, Pair<Double,Integer> second);

    /** Set the final index/result of the accumulation. */
    void setFinalResult( int idx );

    /** Get the final result of the IndexAccumulation */
    int getFinalResult();

    /**Initial value for the index accumulation
     *@return the initial value
     */
    double zeroDouble();

    /** Initial value for the index accumulation.
     * @return the initial value
     * */
    float zeroFloat();

    /** The initial value and initial index to use
     * for the accumulation
     * @return
     */
    Pair<Double,Integer> zeroPair();

    /**Complex initial value
     * @return the complex initial value
     */
    IComplexNumber zeroComplex();
}
