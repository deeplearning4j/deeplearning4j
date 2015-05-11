package org.nd4j.linalg.api.ndarray;


import org.nd4j.linalg.api.ops.impl.transforms.LinearIndex;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Row vector view of an ndarray
 * @author Adam Gibson
 */
public class LinearViewNDArray  extends BaseNDArray {
    private INDArray wrapped;
    private int[] indices;

    public LinearViewNDArray(INDArray wrapped) {
        this.wrapped = wrapped;
        this.shape = new int[] {1,wrapped.length()};
        indices = new int[wrapped.length()];
        LinearIndex index = new LinearIndex(wrapped,wrapped.dup(),true);
        Nd4j.getExecutioner().iterateOverAllRows(index);
        this.indices = index.getIndices();
        this.data = wrapped.data();
        this.offset = wrapped.offset();

    }

    @Override
    public boolean isCleanedUp() {
        return wrapped.isCleanedUp();
    }

    @Override
    public void cleanup() {
        wrapped.cleanup();
    }

    @Override
    public void resetLinearView() {

    }

    @Override
    public int secondaryStride() {
        return wrapped.secondaryStride();
    }

    @Override
    public int majorStride() {
        return wrapped.majorStride();
    }

    @Override
    public INDArray linearView() {
        return this;
    }

    @Override
    public INDArray linearViewColumnOrder() {
        return this;
    }

    @Override
    public int vectorsAlongDimension(int dimension) {
        if(dimension > 1)
            throw new IllegalArgumentException("Linear view does not have dimensions greater than 1");
        return 1;
    }

    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        if(dimension == 0 || dimension == 1 && index == 0)
            return this;
        throw new IllegalArgumentException("Index must be 0 and dimension must be 0 or 1");
    }


    @Override
    public INDArray putScalar(int i, double value) {
        wrapped.data().put(indices[i],value);
        return this;
    }


    @Override
    public int length() {
        return wrapped.length();
    }

    @Override
    public double getDouble(int i) {
        return data.getDouble(indices[i]);
    }


    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("[");
        for(int i = 0; i < wrapped.length(); i++) {
            sb.append(getDouble(i));
            if(i < wrapped.length()  - 1) {
                sb.append(",");
            }
        }

        sb.append("]");
        return sb.toString();
    }

}
