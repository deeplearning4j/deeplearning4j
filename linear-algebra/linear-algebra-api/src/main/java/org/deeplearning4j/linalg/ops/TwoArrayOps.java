package org.deeplearning4j.linalg.ops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.elementwise.AddOp;

/**
 * Builder for two array (possibly plus scalar operations)
 *
 * @author Adam Gibson
 */
public class TwoArrayOps  {


    private INDArray from,to,scalar,other;
    private Class<? extends BaseTwoArrayElementWiseOp> clazz;


    public TwoArrayOps op(Class<? extends BaseTwoArrayElementWiseOp> clazz) {
        this.clazz = clazz;
        return this;
    }


    public TwoArrayOps other(INDArray other) {

        this.other = other;
        return this;
    }

    public TwoArrayOps from(INDArray from) {

        this.from = from;
        return this;
    }

    public TwoArrayOps to(INDArray to) {

        this.to = to;
        return this;
    }

    public TwoArrayOps scalar(INDArray scalar) {
        this.scalar = scalar;
        assert scalar.isScalar() : "Input is not a scalar";
        return this;
    }


    public BaseTwoArrayElementWiseOp build() {
        try {
            BaseTwoArrayElementWiseOp op = clazz.newInstance();
            op.from = from;
            op.to = to;
            op.other = other;
            op.scalarValue = scalar;
            return op;
        }catch (Exception e) {
            throw new RuntimeException(e);

        }
    }


}
