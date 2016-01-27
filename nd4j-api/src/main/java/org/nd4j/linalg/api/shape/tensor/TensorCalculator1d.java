package org.nd4j.linalg.api.shape.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

public class TensorCalculator1d implements TensorCalculator {

    private int baseOffset;
    private int[] shape;
    private int[] stride;
    private int tensorDim;

    private int[] shapeMinusTensorDim;
    private int elementWiseStride;
    private int[] tensorShape;
    private int[] tensorStride;
    private int numTensors;

    public TensorCalculator1d(INDArray arr, int tensorDim) {
        this(arr.offset(), arr.shape(), arr.stride(), tensorDim);
    }

    public TensorCalculator1d(int baseOffset, int[] shape, int[] stride, int tensorDim) {
        this.baseOffset = baseOffset;
        this.shape = shape;
        this.stride = stride;
        //ensure the tensor dimension works properly with the last dimension
        if(tensorDim < 0)
            tensorDim += shape.length;
        this.tensorDim = tensorDim;

        this.shapeMinusTensorDim = ArrayUtil.removeIndex(shape,tensorDim);

        elementWiseStride = stride[tensorDim];
        tensorShape = new int[]{1,shape[tensorDim]};
        tensorStride = new int[]{1,elementWiseStride};

        numTensors = ArrayUtil.prod(shapeMinusTensorDim);
    }

    @Override
    public int getNumTensors() {
        return numTensors;
    }

    public int getOffsetForTensor(int tensorIdx) {
        //Based on: Shape.getOffset()
        int[] indicesMinusTensorDim = Shape.ind2subC(shapeMinusTensorDim, tensorIdx);

        int offset = baseOffset;
        int j=0;
        for( int i=0; i<shape.length; i++ ){
            if(i!=tensorDim){
                offset += indicesMinusTensorDim[j++] * stride[i];
            }
        }

        return offset;
    }

    @Override
    public int[] getShape() {
        return tensorShape;
    }

    @Override
    public int[] getStride() {
        return tensorStride;
    }

    @Override
    public int getBaseOffset() {
        return baseOffset;
    }

    @Override
    public int getElementWiseStrideForTensor(){
        return elementWiseStride;
    }

    @Override
    public int getTensorLength(){
        return shape[tensorDim];
    }
}
