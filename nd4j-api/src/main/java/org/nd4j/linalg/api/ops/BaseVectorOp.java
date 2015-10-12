package org.nd4j.linalg.api.ops;

import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

@NoArgsConstructor
public abstract class BaseVectorOp extends BaseOp implements VectorOp {

    protected int dimension;

    public BaseVectorOp(INDArray x, INDArray y, INDArray z, int dimension){
        super(x,y,z,x.length());
        this.dimension = dimension;
    }

    @Override
    public int getDimension(){
        return dimension;
    }

    @Override
    public void setDimension(int dimension){
        this.dimension = dimension;
    }

    @Override
    public Op opForDimension(int index, int dimension){
        throw new UnsupportedOperationException("opForDimension not supported for VectorOps");
    }

    @Override
    public Op opForDimension(int index, int... dimension){
        throw new UnsupportedOperationException("opForDimension not supported for VectorOps");
    }
}
