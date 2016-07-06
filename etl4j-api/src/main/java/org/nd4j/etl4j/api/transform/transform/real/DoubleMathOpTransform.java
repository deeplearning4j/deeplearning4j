package org.nd4j.etl4j.api.transform.transform.real;

import org.nd4j.etl4j.api.transform.MathOp;
import org.nd4j.etl4j.api.io.data.DoubleWritable;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.DoubleMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnTransform;

/**
 * Double mathematical operation.<br>
 * This is an in-place operation of the double column value and a double scalar.
 *
 * @author Alex Black
 * @see DoubleColumnsMathOpTransform to do a mathematical operation involving multiple columns (instead of a scalar)
 */
public class DoubleMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final double scalar;

    public DoubleMathOpTransform(String columnName, MathOp mathOp, double scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof DoubleMetaData))
            throw new IllegalStateException("Column is not an integer column");
        DoubleMetaData meta = (DoubleMetaData) oldColumnType;
        Double minValue = meta.getMin();
        Double maxValue = meta.getMax();
        if (minValue != null) minValue = doOp(minValue);
        if (maxValue != null) maxValue = doOp(maxValue);
        if(minValue != null && maxValue != null && minValue > maxValue ){
            //Consider rsub 1, with original min/max of 0 and 1: (1-0) -> 1 and (1-1) -> 0
            //Or multiplication by -1: (0 to 1) -> (-1 to 0)
            //Need to swap min/max here...
            Double temp = minValue;
            minValue = maxValue;
            maxValue = temp;
        }
        return new DoubleMetaData(minValue, maxValue);
    }

    private double doOp(double input) {
        switch (mathOp) {
            case Add:
                return input + scalar;
            case Subtract:
                return input - scalar;
            case Multiply:
                return input * scalar;
            case Divide:
                return input / scalar;
            case Modulus:
                return input % scalar;
            case ReverseSubtract:
                return scalar - input;
            case ReverseDivide:
                return scalar / input;
            case ScalarMin:
                return Math.min(input, scalar);
            case ScalarMax:
                return Math.max(input, scalar);
            default:
                throw new IllegalStateException("Unknown or not implemented math op: " + mathOp);
        }
    }

    @Override
    public Writable map(Writable columnWritable) {
        return new DoubleWritable(doOp(columnWritable.toDouble()));
    }

    @Override
    public String toString() {
        return "DoubleMathOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
    }
}
