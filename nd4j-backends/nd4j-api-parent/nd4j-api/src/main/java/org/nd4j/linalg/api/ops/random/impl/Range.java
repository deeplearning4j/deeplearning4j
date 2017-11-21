package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Range Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Range extends BaseRandomOp {
    private double from;
    private double to;

    public Range() {
        // no-op
    }

    public Range(double from, double to, int length) {
        this(Nd4j.createUninitialized(new int[] {1, length}, Nd4j.order()), from, to);
    }

    public Range(@NonNull INDArray z, double from, double to) {
        this.from = from;
        this.to = to;
        init(null, null, z, z.lengthLong());
        this.extraArgs = new Object[] {from, to};
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "range";
    }

    @Override
    public String onnxName() {
        return "Range";
    }

    @Override
    public String tensorflowName() {
        return "Range";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to differentiate array creation routine");
    }
}
