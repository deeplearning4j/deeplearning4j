package org.nd4j.linalg.api.parallel.bufferops;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;

import java.util.concurrent.RecursiveAction;

/** A DataBufferAction for executing Scalar ops on a DataBuffer in parallel
 * @author Alex Black
 */
public abstract class ScalarDataBufferAction extends RecursiveAction {
    protected final ScalarOp op;
    protected final int threshold;
    protected int n;
    protected final DataBuffer x;
    protected final DataBuffer z;
    protected int offsetX;
    protected int offsetZ;
    protected int incrX;
    protected int incrZ;

    protected final boolean doTensorFirst;
    protected INDArray ndx;
    protected INDArray ndz;
    protected int tadIdx;
    protected int tadDim;


    public ScalarDataBufferAction(ScalarOp op, int threshold, int n, DataBuffer x, DataBuffer z,
                                  int offsetX, int offsetZ, int incrX, int incrZ){
        this.op = op;
        this.threshold = threshold;
        this.n = n;
        this.x = x;
        this.z = z;
        this.offsetX = offsetX;
        this.offsetZ = offsetZ;
        this.incrX = incrX;
        this.incrZ = incrZ;
        this.doTensorFirst = false;
    }


    public ScalarDataBufferAction(ScalarOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray z){
        this.op = op;
        this.threshold = threshold;
        this.x = x.data();
        this.z = z.data();
        this.ndx = x;
        this.ndz = z;
        this.tadIdx = tadIdx;
        this.tadDim = tadDim;
        this.doTensorFirst = true;
    }

    @Override
    protected void compute() {
        if(doTensorFirst){
            INDArray tadX = ndx.tensorAlongDimension(tadIdx, tadDim);
            INDArray tadZ = (ndz != ndx ? ndz.tensorAlongDimension(tadIdx, tadDim) : tadX);
            this.offsetX = tadX.offset();
            this.offsetZ = tadZ.offset();
            this.incrX = tadX.elementWiseStride();
            this.incrZ = tadZ.elementWiseStride();
            this.n = tadX.length();
        }
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
