package org.nd4j.linalg.api.ops;

import lombok.AllArgsConstructor;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ops.Op;


public interface IndexAccumulation extends Op {

    int update(double accum, int accumIdx, double x, int xIdx);

    int update(float accum, int accumIdx, float x, int xIdx);

    int update(double accum, int accumIdx, double x, double y, int idx);

    int update(float accum, int accumIdx, float x, float y, int idx);

    int update(IComplexNumber accum, int accumIdx, IComplexNumber x, int xIdx);

    int update(IComplexNumber accum, int accumIdx, double x, int idx);

    int update(IComplexNumber accum, int accumIdx, double x, double y, int idx);

    int update(IComplexNumber accum, int accumIdx, IComplexNumber x, IComplexNumber y, int idx);

    int combineSubResults(double first, int idxFirst, double second, int idxSecond);

    int combineSubResults(float first, int idxFirst, float second, int idxSecond);

    Pair<Double,Integer> combineSubResults(Pair<Double,Integer> first, Pair<Double,Integer> second);

    void setFinalResult( int idx );

    /**Initial value
     *@return the initial value
     */
    double zeroDouble();

    float zeroFloat();

    Pair<Double,Integer> zeroPair();

    /**Complex initial value
     * @return the complex initial value
     */
    IComplexNumber zeroComplex();
}
