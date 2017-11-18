package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Data
public class Constant extends BaseTransformOp {

    protected SDVariable m_x;
    protected int[] shape;

    public Constant() {
    }


    protected Constant(SameDiff sameDiff,
                       SDVariable i_v,
                       int[] shape,
                       boolean inPlace,int[] vertexId) {
        super();
        this.shape = shape;
        this.inPlace = inPlace;
        this.sameDiff = sameDiff;
        this.args = new DifferentialFunction[] {this};
        if (i_v != null) {
            m_x = i_v;

        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        this.vertexId = vertexId;
        f().validateFunctionReference(this);
        if(sameDiff.getGraph().getVertex(this.vertexId[0]) == null) {
            sameDiff.getGraph().addVertex(new NDArrayVertex(sameDiff,vertexId[0],0,i_v));
        }

    }

    public Constant(SameDiff sameDiff,
                    SDVariable i_v,
                    int[] shape,int[] vertexId) {
        this(sameDiff,i_v,shape,false,vertexId);
    }

    public Constant(INDArray x, INDArray z, int[] shape) {
        super(x, z);
        this.shape = shape;
    }

    public Constant(int[] shape) {
        this.shape = shape;
    }

    public Constant(INDArray x, INDArray z, long n, int[] shape) {
        super(x, z, n);
        this.shape = shape;
    }

    public Constant(INDArray x, INDArray y, INDArray z, long n, int[] shape) {
        super(x, y, z, n);
        this.shape = shape;
    }

    public Constant(INDArray x, int[] shape) {
        super(x);
        this.shape = shape;
    }

    /**
     * Get the result shape for this function
     *
     * @return
     */
    @Override
    public int[] getResultShape() {
        return shape;
    }

    @Override
    public boolean isConstant() {
        return true;
    }


    @Override
    public SDVariable getResult() {
        return this.m_x;
    }

    @Override
    public DifferentialFunction[] args() {
        return new DifferentialFunction[] {this};
    }

    @Override
    public DifferentialFunction arg() {
        return this;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        f().validateDifferentialFunctionsameDiff(i_v);
        return Collections.<DifferentialFunction> singletonList(sameDiff.zero("grad-" + UUID.randomUUID().toString(),i_v.get(0).getShape()));

    }

    @Override
    public String toString() {
        return m_x.toString();
    }



    @Override
    public DifferentialFunction dup() {
        Constant ret = sameDiff.setupFunction(new Constant(sameDiff, m_x, shape,vertexId));
        Constant differentialFunction = ret;
        return differentialFunction;
    }



    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "constant";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
      throw new NoOpNameFoundException("No tensorflow opName found for " + calculateOutputShape());
    }

}
