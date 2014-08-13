package org.deeplearning4j.linalg.ops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Builder for element wise operations
 *
 * @author Adam Gibson
 */
public class ArrayOps {

    private INDArray from,scalar;
    private Class<BaseElementWiseOp> clazz;


    public ArrayOps op(Class<BaseElementWiseOp> clazz) {
        this.clazz = clazz;
        return this;
    }


    public ArrayOps from(INDArray from) {
        this.from = from;
        return this;
    }


    public ArrayOps scalar(INDArray scalar) {
        this.scalar = scalar;
        return this;
    }


    public ElementWiseOp build() {
        try {
            BaseElementWiseOp op = clazz.newInstance();
            op.from = from;
            op.scalarValue = scalar;
            return op;
        }catch (Exception e) {
            throw new RuntimeException(e);

        }
    }

}
