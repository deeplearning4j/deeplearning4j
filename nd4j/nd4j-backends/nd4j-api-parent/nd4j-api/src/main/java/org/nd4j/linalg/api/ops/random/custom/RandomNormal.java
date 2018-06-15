package org.nd4j.linalg.api.ops.random.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * This op is a wrapper for RandomNormal Op
 * @author raver119@gmail.com
 */
public class RandomNormal extends DynamicCustomOp {

    private double mean;
    private double stdev;

    public RandomNormal() {

    }

    public RandomNormal(SameDiff sameDiff, SDVariable shape, double mean, double stdev) {
        super(null, sameDiff, new SDVariable[]{shape});
        this.mean = mean;
        this.stdev = stdev;

        addTArgument(mean, stdev);
        throw new UnsupportedOperationException("Disabled pending TF import fix (duplicate names)");
    }

    @Override
    public String opName() {
        return "randomnormal";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Not TF op name set for " + getClass().getSimpleName());
    }
}
