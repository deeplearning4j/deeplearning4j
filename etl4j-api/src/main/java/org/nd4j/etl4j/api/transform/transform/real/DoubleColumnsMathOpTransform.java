package org.nd4j.etl4j.api.transform.transform.real;

import org.nd4j.etl4j.api.transform.MathOp;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.DoubleMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnsMathOpTransform;
import org.nd4j.etl4j.api.io.data.DoubleWritable;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Add a new double column, calculated from one or more other columns. A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==Add, and columns=={"col1","col2"}, then the output column
 * with name "newCol" has value col1+col2.<br>
 *
 * @author Alex Black
 * @see DoubleMathOpTransform To do an in-place mathematical operation of a double column and a double scalar value
 */
public class DoubleColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    public DoubleColumnsMathOpTransform(String newColumnName, MathOp mathOp, String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData() {
        return new DoubleMetaData();
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                double sum = 0;
                for (Writable w : input) sum += w.toDouble();
                return new DoubleWritable(sum);
            case Subtract:
                return new DoubleWritable(input[0].toDouble() - input[1].toDouble());
            case Multiply:
                double product = 1.0;
                for (Writable w : input) product *= w.toDouble();
                return new DoubleWritable(product);
            case Divide:
                return new DoubleWritable(input[0].toDouble() / input[1].toDouble());
            case Modulus:
                return new DoubleWritable(input[0].toDouble() % input[1].toDouble());
            case ReverseSubtract:
            case ReverseDivide:
            case ScalarMin:
            case ScalarMax:
            default:
                throw new RuntimeException("Invalid mathOp: " + mathOp);    //Should never happen
        }
    }
}
