package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;

import java.util.concurrent.RecursiveAction;

@AllArgsConstructor
public abstract class TransformDataBufferAction extends RecursiveAction {
    protected final TransformOp op;
    protected final int threshold;
    protected final int n;
    protected final DataBuffer x;
    protected final DataBuffer y;
    protected final DataBuffer z;
    protected final int offsetX;
    protected final int offsetY;
    protected final int offsetZ;
    protected final int incrX;
    protected final int incrY;
    protected final int incrZ;

    /**
     * Constructor for doing a 1d TAD first.
     */
    public TransformDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        this.op = op;
        INDArray tadX = x.tensorAlongDimension(tadIdx, tadDim);
        INDArray tadY = (y != null ? y.tensorAlongDimension(tadIdx, tadDim) : null);
        INDArray tadZ = (z != x ? z.tensorAlongDimension(tadIdx, tadDim) : tadX);
        this.x = x.data();
        this.y = (y != null ? y.data() : null);
        this.z = z.data();
        this.offsetX = tadX.offset();
        this.offsetY = (y != null ? tadY.offset() : 0);
        this.offsetZ = tadZ.offset();
        this.incrX = tadX.elementWiseStride();
        this.incrY = (tadY != null ? tadY.elementWiseStride() : 0);
        this.incrZ = tadZ.elementWiseStride();
        this.threshold = threshold;
        this.n = tadX.length();
    }

    @Override
    protected void compute() {
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
