package org.nd4j.linalg.api.parallel.bufferops.impl.transform;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.bufferops.TransformDataBufferAction;

public class MulOpDataBufferAction extends TransformDataBufferAction {

    public MulOpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public MulOpDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        super(op, tadIdx, tadDim, threshold, x, y, z);
    }

    @Override
    public void doTask() {
        //Task: Z = X*Y

        if (x.dataType() == DataBuffer.Type.FLOAT) {
            float[] xf = (float[]) x.array();
            float[] yf = (float[]) y.array();
            if (incrX == 1 && incrY == 1 && incrZ == 1) {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        xf[offsetX + i] *= yf[offsetY + i];
                    }
                } else {
                    float[] zf = (float[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zf[offsetZ + i] = xf[offsetX + i] * yf[offsetY + i];
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        xf[offsetX + i * incrX] *= yf[offsetY + i * incrY];
                    }
                } else {
                    float[] zf = (float[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] * yf[offsetY + i * incrY];
                    }
                }
            }
        } else {
            double[] xd = (double[]) x.array();
            double[] yd = (double[]) y.array();
            if (incrX == 1 && incrY == 1 && incrZ == 1) {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        xd[offsetX + i] *= yd[offsetY + i];
                    }
                } else {
                    double[] zd = (double[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zd[offsetZ + i] = xd[offsetX + i] * yd[offsetY + i];
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        xd[offsetX + i * incrX] *= yd[offsetY + i * incrY];
                    }
                } else {
                    double[] zd = (double[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] * yd[offsetY + i * incrY];
                    }
                }
            }
        }
    }

    @Override
    public TransformDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        return new MulOpDataBufferAction(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }
}
