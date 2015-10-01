package org.nd4j.linalg.api.parallel.bufferops;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;

import java.util.concurrent.RecursiveAction;

public abstract class TransformDataBufferAction extends RecursiveAction {
    protected final TransformOp op;
    protected final int threshold;
    protected int n;
    protected final DataBuffer x;
    protected final DataBuffer y;
    protected final DataBuffer z;
    protected int offsetX;
    protected int offsetY;
    protected int offsetZ;
    protected int incrX;
    protected int incrY;
    protected int incrZ;

    protected final boolean doTensorFirst;
    protected INDArray ndx;
    protected INDArray ndy;
    protected INDArray ndz;
    protected int tadIdx;
    protected int tadDim;

    public TransformDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z,
                                     int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ ){
        this.op = op;
        this.threshold = threshold;
        this.n = n;
        this.x = x;
        this.y = y;
        this.z = z;
        this.offsetX = offsetX;
        this.offsetY = offsetY;
        this.offsetZ = offsetZ;
        this.incrX = incrX;
        this.incrY = incrY;
        this.incrZ = incrZ;
        this.doTensorFirst = false;
    }

    /**
     * Constructor for doing a 1d TAD first.
     */
    public TransformDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        this.op = op;
        this.tadIdx = tadIdx;
        this.tadDim = tadDim;
        this.x = x.data();
        this.y = (y!=null ? y.data() : null);
        this.z = z.data();
        this.threshold = threshold;
        this.doTensorFirst = true;
    }

    @Override
    protected void compute() {
        if(doTensorFirst){
            INDArray tadX = ndx.tensorAlongDimension(tadIdx, tadDim);
            INDArray tadY = (ndy != null ? ndy.tensorAlongDimension(tadIdx, tadDim) : null);
            INDArray tadZ = (ndz != ndx ? ndz.tensorAlongDimension(tadIdx, tadDim) : tadX);
            this.offsetX = tadX.offset();
            this.offsetY = (y != null ? tadY.offset() : 0);
            this.offsetZ = tadZ.offset();
            this.incrX = tadX.elementWiseStride();
            this.incrY = (tadY != null ? tadY.elementWiseStride() : 0);
            this.incrZ = tadZ.elementWiseStride();
            this.n = tadX.length();
        }

        if (n > threshold) {
            //Split task
            int nFirst = n / 2;
            TransformDataBufferAction t1 = getSubTask(threshold, nFirst, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            t1.fork();

            int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            int offsetZ2 = offsetZ + nFirst * incrZ;
            TransformDataBufferAction t2 = getSubTask(threshold, nSecond, x, y, z, offsetX2, offsetY2, offsetZ2, incrX, incrY, incrZ);
            t2.fork();

            t1.join();
            t2.join();
        } else {
            doTask();
        }
    }

    public abstract void doTask();

    public abstract TransformDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z,
                                                         int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ);

}
