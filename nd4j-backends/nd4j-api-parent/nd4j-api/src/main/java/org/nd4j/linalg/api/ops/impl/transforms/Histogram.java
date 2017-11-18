package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class Histogram extends BaseTransformOp {
    public Histogram(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int numBins) {
        super(sameDiff, i_v, inPlace);
        this.numBins = numBins;
    }

    public Histogram(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs, int numBins) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.numBins = numBins;
    }

    public Histogram(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs, int numBins) {
        super(sameDiff, i_v, extraArgs);
        this.numBins = numBins;
    }

    private int numBins = 0;

    public Histogram() {
        //no-op
    }

    public Histogram(INDArray x, INDArray z) {
        setX(x);
        setZ(z);
        numBins = z.length();

        double max = x.maxNumber().doubleValue();
        double min = x.minNumber().doubleValue();

        this.extraArgs = new Object[] {(double) numBins, min, max};
    }

    public Histogram(INDArray x, int numberOfBins) {
        this(x, Nd4j.create(numberOfBins));
    }

    @Override
    public int opNum() {
        return 48;
    }

    @Override
    public String opName() {
        return "histogram";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "histogram";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
