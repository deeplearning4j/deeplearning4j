package org.datavec.api.transform.transform.longtransform;

import org.datavec.api.io.data.LongWritable;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.LongMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Writable;

/**
 * Long mathematical operation.<br>
 * This is an in-place operation of the long column value and an long scalar.
 *
 * @author Alex Black
 * @see LongColumnsMathOpTransform to do a mathematical operation involving multiple long columns (instead of a scalar)
 */
public class LongMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final long scalar;

    public LongMathOpTransform(String columnName, MathOp mathOp, long scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof LongMetaData))
            throw new IllegalStateException("Column is not an Long column");
        LongMetaData meta = (LongMetaData) oldColumnType;
        Long minValue = meta.getMin();
        Long maxValue = meta.getMax();
        if (minValue != null) minValue = doOp(minValue);
        if (maxValue != null) maxValue = doOp(maxValue);
        if(minValue != null && maxValue != null && minValue > maxValue ){
            //Consider rsub 1, with original min/max of 0 and 1: (1-0) -> 1 and (1-1) -> 0
            //Or multiplication by -1: (0 to 1) -> (-1 to 0)
            //Need to swap min/max here...
            Long temp = minValue;
            minValue = maxValue;
            maxValue = temp;
        }
        return new LongMetaData(minValue, maxValue);
    }

    private long doOp(long input) {
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
        return new LongWritable(doOp(columnWritable.toLong()));
    }

    @Override
    public String toString() {
        return "LongMathOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
    }
}
