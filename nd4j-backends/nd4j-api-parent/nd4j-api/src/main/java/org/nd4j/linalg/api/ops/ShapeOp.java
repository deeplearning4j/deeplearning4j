package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Shape manipulation ops
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class ShapeOp extends BaseOp {
    public ShapeOp() {}





    public ShapeOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }



    public ShapeOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        this(sameDiff,i_v,i_v.getShape(),inPlace,null);
    }

    public ShapeOp(SameDiff sameDiff,
                   SDVariable i_v,
                   int[] shape,
                   boolean inPlace,
                   Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);

        if (i_v != null) {
            f().validateDifferentialFunctionsameDiff(i_v);
            val vertexId = outputVariables()[0].getVertexId();
            sameDiff.putShapeForVertexId(vertexId,shape);
            sameDiff.addArgsFor(new SDVariable[] {i_v},this);
            sameDiff.addOutgoingFor(new int[]{vertexId},this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }
    /**
     * Specify an alternative output array
     *
     * @param x the input
     * @param z the output
     * @param n the number of elements to iterate on
     */
    public ShapeOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public ShapeOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>();
        val vertexId = outputVariables()[0].getVertexId();
        ret.add(sameDiff.getShapeForVertexId(vertexId));
        return ret;
    }

    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        super.initWithArrays(arrayMap);
        val shapeOutput = calculateOutputShape();
        val vertexId = outputVariables()[0].getVertexId();
        if(!shapeOutput.isEmpty() && sameDiff.shapeAlreadyExistsForVertexId(vertexId))
            sameDiff.updateShapeForVertexId(vertexId,shapeOutput.get(0));
        else if(!shapeOutput.isEmpty())
            sameDiff.putShapeForVertexId(vertexId,shapeOutput.get(0));

    }

    @Override
    public Type opType() {
        return Type.SHAPE;
    }

    /**
     * An op for one ndarray
     *
     * @param x the ndarray
     */
    public ShapeOp(INDArray x) {
        super(x);
    }

    /**
     * Specify an alternative result array
     *
     * @param x the input
     * @param z the output array
     */
    public ShapeOp(INDArray x, INDArray z) {
        super(x, z);
    }





}
