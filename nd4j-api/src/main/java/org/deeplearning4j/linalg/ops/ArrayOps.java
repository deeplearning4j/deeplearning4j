package org.deeplearning4j.linalg.ops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.ReflectionUtil;

import java.lang.reflect.Constructor;

/**
 * Builder for element wise operations
 *
 * @author Adam Gibson
 */
public class ArrayOps {

    private INDArray from,scalar;
    private Class<? extends ElementWiseOp> clazz;
    private Object[] extraArgs;


    /**
     * Extra arguments for a constructor
     * @param extraArgs the extra arguments for a constructor
     * @return
     */
    public ArrayOps extraArgs(Object[] extraArgs) {
        this.extraArgs = extraArgs;
        return this;
    }

    /**
     * The operation to perform
     * @param clazz the class of the operation to perform
     * @return builder pattern
     */
    public ArrayOps op(Class<? extends ElementWiseOp> clazz) {
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
            ElementWiseOp op;
            if(extraArgs == null)
                op = clazz.newInstance();
            else {
                Constructor c = clazz.getConstructor(ReflectionUtil.classesFor(extraArgs));
                op = (ElementWiseOp) c.newInstance(extraArgs);
            }
            BaseElementWiseOp op2 = (BaseElementWiseOp) op;
            op2.from = from;
            op2.scalarValue = scalar;
            return op;
        }catch (Exception e) {
            throw new RuntimeException(e);

        }
    }

}
