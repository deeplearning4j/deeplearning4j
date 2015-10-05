package org.nd4j.linalg.api.parallel.bufferops.impl.transform;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.bufferops.TransformDataBufferAction;
import org.nd4j.linalg.factory.Nd4j;

public class CopyOpDataBufferAction extends TransformDataBufferAction {
    public CopyOpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public CopyOpDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        super(op, tadIdx, tadDim, threshold, x, y, z);
    }

    @Override
    public void doTask() {
        //Task: Z = X
        if (x == z) return;    //No op
        if(incrX == 1 && incrZ == 1 && x.allocationMode() == DataBuffer.AllocationMode.HEAP){
            if(x.dataType() == DataBuffer.Type.FLOAT){
                float[] fx = (float[])x.array();
                float[] fz = (float[])z.array();
                System.arraycopy(fx,offsetX,fz,offsetZ,n);  //z = x
            } else {
                double[] dx = (double[])x.array();
                double[] dz = (double[])z.array();
                System.arraycopy(dx,offsetX,dz,offsetZ,n);  //z = x
            }
            return;
        }
        Nd4j.getBlasWrapper().level1().copy(n, x, offsetX, incrX, z, offsetZ, incrZ);
    }

    @Override
    public TransformDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        return new CopyOpDataBufferAction(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }
}
