package org.nd4j.linalg.api.ops.exception;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class BlasOpErrorMessage implements Serializable {
    private Op op;

    public BlasOpErrorMessage(Op op) {
        this.op = op;
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer().append("Op " + op.name() + " of length " + op.n()).append(" will fail with x of " + shapeInfo(op.x()));
        if(op.y() != null) {
            sb.append(" y of " + shapeInfo(op.y()));
        }

        sb.append(" and z of " + op.z());
        return sb.toString();
    }

    private String shapeInfo(INDArray arr) {
        return Arrays.toString(arr.shape()) + " and stride " + Arrays.toString(arr.stride()) + " and offset " + arr.offset() + " and blas stride of " + BlasBufferUtil.getBlasStride(arr);
    }


}
