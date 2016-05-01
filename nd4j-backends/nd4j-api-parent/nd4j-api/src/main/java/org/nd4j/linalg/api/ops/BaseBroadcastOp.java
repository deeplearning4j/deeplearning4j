package org.nd4j.linalg.api.ops;

import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

@NoArgsConstructor
public abstract class BaseBroadcastOp extends BaseOp implements BroadcastOp {

    protected int[] dimension;

    public BaseBroadcastOp(INDArray x, INDArray y, INDArray z, int...dimension) {
        super(x,y,z,x.lengthLong());
        this.dimension = dimension;
        for(int i = 0; i < dimension.length; i++)
            if(dimension[i] < 0)
                dimension[i] += x.rank();
        if(y.length() != x.size(dimension[0])) {
            throw new IllegalArgumentException("Unable to broadcast y along dimension " + dimension[0] + " dimension must be same length");
        }
    }



    @Override
    public int broadcastLength() {
        if(y == null)
            throw new IllegalStateException("Unable to get broad cast length for y, no y specified");
        return y.length();
    }

    @Override
    public int[] broadcastShape() {
        if(y == null)
            throw new IllegalStateException("Unable to get broad cast shape for y, no y specified");
        return y.shape();
    }

    @Override
    public int[] getDimension(){
        return dimension;
    }

    @Override
    public void setDimension(int...dimension){
        this.dimension = dimension;
    }

    @Override
    public Op opForDimension(int index, int dimension){
        throw new UnsupportedOperationException("opForDimension not supported for BroadcastOps");
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        throw new UnsupportedOperationException("opForDimension not supported for BroadcastOps");
    }
}
