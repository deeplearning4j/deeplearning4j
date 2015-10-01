package org.nd4j.linalg.api.parallel.bufferops.impl.scalar;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.parallel.bufferops.ScalarDataBufferAction;

public class ScalarOpDataBufferAction extends ScalarDataBufferAction {

    public ScalarOpDataBufferAction(ScalarOp op, int threshold, int n, DataBuffer x, DataBuffer z, int offsetX, int offsetZ,
                                    int incrX, int incrZ) {
        super(op, threshold, n, x, z, offsetX, offsetZ, incrX, incrZ);
    }

    public ScalarOpDataBufferAction(ScalarOp op, int tensorNum, int tensorDim, int threshold, INDArray x, INDArray z){
        super(op, tensorNum,tensorDim,threshold,x,null,z);
    }

    @Override
    public void doTask() {
        //Task: Z = Op(X)
        if (x.dataType() == DataBuffer.Type.FLOAT) {
            float[] xf = (float[]) x.array();
            if (incrX == 1 && incrZ == 1) {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i;
                        xf[xIdx] = op.op(xf[xIdx]);
                    }
                } else {
                    float[] zf = (float[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zf[offsetZ + i] = op.op(xf[offsetX + i]);
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i * incrX;
                        xf[xIdx] = op.op(xf[xIdx]);
                    }
                } else {
                    float[] zf = (float[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX]);
                    }
                }
            }
        } else {
            double[] xd = (double[]) x.array();
            if (incrX == 1 && incrZ == 1) {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i;
                        xd[xIdx] = op.op(xd[xIdx]);
                    }
                } else {
                    double[] zd = (double[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zd[offsetZ + i] = op.op(xd[offsetX + i]);
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i * incrX;
                        xd[xIdx] = op.op(xd[xIdx]);
                    }
                } else {
                    double[] zd = (double[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX]);
                    }
                }
            }
        }
    }

    @Override
    public ScalarDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer z, int offsetX, int offsetZ,
                                             int incrX, int incrZ) {
        return new ScalarOpDataBufferAction(op, threshold, n, x, z, offsetX, offsetZ, incrX, incrZ);
    }
}
