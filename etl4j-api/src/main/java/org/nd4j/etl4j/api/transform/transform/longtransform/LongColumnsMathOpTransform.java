package org.nd4j.etl4j.api.transform.transform.longtransform;

import org.nd4j.etl4j.api.transform.MathOp;
import org.nd4j.etl4j.api.transform.metadata.LongMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnsMathOpTransform;
import org.nd4j.etl4j.api.transform.transform.real.DoubleColumnsMathOpTransform;
import org.nd4j.etl4j.api.io.data.LongWritable;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;

/**
 * Add a new long column, calculated from one or more other columns. A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==MathOp.Add, and columns=={"col1","col2"}, then the output column
 * with name "newCol" has value col1+col2.<br>
 * <b>NOTE</b>: Division here is using long division (long output). Use {@link DoubleColumnsMathOpTransform}
 * if a decimal output value is required.
 *
 * @author Alex Black
 * @see LongMathOpTransform To do an in-place mathematical operation of a long column and a long scalar value
 */
public class LongColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    public LongColumnsMathOpTransform(String newColumnName, MathOp mathOp, String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData() {
        return new LongMetaData();
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                long sum = 0;
                for (Writable w : input) sum += w.toLong();
                return new LongWritable(sum);
            case Subtract:
                return new LongWritable(input[0].toLong() - input[1].toLong());
            case Multiply:
                long product = 1;
                for (Writable w : input) product *= w.toLong();
                return new LongWritable(product);
            case Divide:
                return new LongWritable(input[0].toLong() / input[1].toLong());
            case Modulus:
                return new LongWritable(input[0].toLong() % input[1].toLong());
            case ReverseSubtract:
            case ReverseDivide:
            case ScalarMin:
            case ScalarMax:
            default:
                throw new RuntimeException("Invalid mathOp: " + mathOp);    //Should never happen
        }
    }
}
