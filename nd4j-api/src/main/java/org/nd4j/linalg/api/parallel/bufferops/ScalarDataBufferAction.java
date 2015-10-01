package org.nd4j.linalg.api.parallel.bufferops;


import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;

import java.util.concurrent.RecursiveAction;

@AllArgsConstructor
public abstract class ScalarDataBufferAction extends RecursiveAction {
    protected final ScalarOp op;
    protected final int threshold;
    protected final int n;
    protected final DataBuffer x;
    protected final DataBuffer z;
    protected final int offsetX;
    protected final int offsetZ;
    protected final int incrX;
    protected final int incrZ;

    public ScalarDataBufferAction(ScalarOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z){
        this.op = op;
        INDArray tadX = x.tensorAlongDimension(tadIdx, tadDim);
        INDArray tadZ;
        if(z==null) tadZ = null;
        else tadZ = (z != x ? z.tensorAlongDimension(tadIdx, tadDim) : tadX);
        this.x = x.data();
        this.z = (z!=null ? z.data() : null);
        this.offsetX = tadX.offset();
        this.offsetZ = (z!=null ? tadZ.offset() : 0);
        this.incrX = tadX.elementWiseStride();
        this.incrZ = (z!=null ? tadZ.elementWiseStride() : 0);
        this.threshold = threshold;
        this.n = tadX.length();
    }

    @Override
    protected void compute() {
        if (n > threshold) {
            //Split task
            int nFirst = n / 2;
            ScalarDataBufferAction t1 = getSubTask(threshold, nFirst, x, z, offsetX, offsetZ, incrX, incrZ);
            t1.fork();

            int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetZ2 = offsetZ + nFirst * incrZ;
            ScalarDataBufferAction t2 = getSubTask(threshold, nSecond, x, z, offsetX2, offsetZ2, incrX, incrZ);
            t2.fork();

            t1.join();
            t2.join();
        } else {
            doTask();
        }
    }

    public abstract void doTask();


    public abstract ScalarDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer z,
                                                      int offsetX, int offsetZ, int incrX, int incrZ);
}
