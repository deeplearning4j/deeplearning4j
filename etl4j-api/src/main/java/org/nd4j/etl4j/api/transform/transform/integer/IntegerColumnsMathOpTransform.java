package org.nd4j.etl4j.api.transform.transform.integer;

import io.skymind.echidna.api.MathOp;
import org.nd4j.etl4j.api.transform.metadata.IntegerMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnsMathOpTransform;
import org.nd4j.etl4j.api.transform.transform.real.DoubleColumnsMathOpTransform;
import org.canova.api.io.data.IntWritable;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;

/**
 * Add a new integer column, calculated from one or more other columns. A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==MathOp.Add, and columns=={"col1","col2"}, then the output column
 * with name "newCol" has value col1+col2.<br>
 * <b>NOTE</b>: Division here is using integer division (integer output). Use {@link DoubleColumnsMathOpTransform}
 * if a decimal output value is required.
 *
 * @author Alex Black
 * @see IntegerMathOpTransform To do an in-place mathematical operation of an integer column and an integer scalar value
 */
public class IntegerColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    /**
     * @param newColumnName Name of the new column (output column)
     * @param mathOp        Mathematical operation. Only Add/Subtract/Multiply/Divide/Modulus is allowed here
     * @param columns       Columns to use in the mathematical operation
     */
    public IntegerColumnsMathOpTransform(String newColumnName, MathOp mathOp, String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData() {
        return new IntegerMetaData();
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                int sum = 0;
                for (Writable w : input) sum += w.toInt();
                return new IntWritable(sum);
            case Subtract:
                return new IntWritable(input[0].toInt() - input[1].toInt());
            case Multiply:
                int product = 1;
                for (Writable w : input) product *= w.toInt();
                return new IntWritable(product);
            case Divide:
                return new IntWritable(input[0].toInt() / input[1].toInt());
            case Modulus:
                return new IntWritable(input[0].toInt() % input[1].toInt());
            case ReverseSubtract:
            case ReverseDivide:
            case ScalarMin:
            case ScalarMax:
            default:
                throw new RuntimeException("Invalid mathOp: " + mathOp);    //Should never happen
        }
    }
}
