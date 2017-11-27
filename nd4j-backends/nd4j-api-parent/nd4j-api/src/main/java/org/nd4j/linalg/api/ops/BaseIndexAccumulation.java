package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * Index based reduction algo
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseIndexAccumulation extends BaseOp implements IndexAccumulation {
    protected int finalResult;


    public BaseIndexAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[]{i_v},this);
            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }

    public BaseIndexAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            DifferentialFunction i_v2,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {i_v,i_v2},this);
            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            f().validateDifferentialFunctionsameDiff(i_v2);
            addAsNewVertexId();
            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }


    public BaseIndexAccumulation() {}

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseIndexAccumulation(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init();
    }

    public BaseIndexAccumulation(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public BaseIndexAccumulation(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    public BaseIndexAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.lengthLong());
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public float zeroFloat() {
        return 0.0f;
    }

    @Override
    public Pair<Double, Integer> zeroPair() {
        return new Pair<>(zeroDouble(), -1);
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    private void init() {
        init(x, y, x, x.lengthLong());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            this.extraArgs = new Object[] {zeroDouble()};
        } else if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            this.extraArgs = new Object[] {zeroFloat()};
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            this.extraArgs = new Object[] {zeroHalf()};
        }
    }


    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(Shape.getReducedShape(arg().getResultShape(),dimensions));
        return ret;
    }



    @Override
    public void setFinalResult(int idx) {
        this.finalResult = idx;
    }

    @Override
    public int getFinalResult() {
        return finalResult;
    }


}
