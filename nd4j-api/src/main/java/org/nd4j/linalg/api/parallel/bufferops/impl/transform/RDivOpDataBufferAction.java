package org.nd4j.linalg.api.parallel.bufferops.impl.transform;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.bufferops.TransformDataBufferAction;

public class RDivOpDataBufferAction extends TransformDataBufferAction {

    public RDivOpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public RDivOpDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        super(op, tadIdx, tadDim, threshold, x, y, z);
    }

    @Override
    public void doTask() {
        //Task: Z = Y/X

        if (x.dataType() == DataBuffer.Type.FLOAT) {
            float[] xf = (float[]) x.array();
            float[] yf = (float[]) y.array();
            if (incrX == 1 && incrY == 1 && (x==z || incrZ == 1) ) {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX+i;
                        xf[xIdx] = yf[offsetY + i] / xf[xIdx];
                    }
                } else {
                    float[] zf = (float[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zf[offsetZ + i] = yf[offsetY + i] / xf[offsetX + i];
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i * incrX;
                        xf[xIdx] = yf[offsetY + i * incrY] / xf[xIdx];
                    }
                } else {
                    float[] zf = (float[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zf[offsetZ + i * incrZ] = yf[offsetY + i * incrY] / xf[offsetX + i * incrX];
                    }
                }
            }
        } else {
            double[] xd = (double[]) x.array();
            double[] yd = (double[]) y.array();
            if (incrX == 1 && incrY == 1 && (x==z || incrZ == 1)) {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i;
                        xd[xIdx] = yd[offsetY + i] / xd[xIdx];
                    }
                } else {
                    double[] zd = (double[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zd[offsetZ + i] = yd[offsetY + i] / xd[offsetX + i];
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < n; i++) {
                        int xIdx = offsetX + i * incrX;
                        xd[xIdx] = yd[offsetY + i * incrY] / xd[xIdx];
                    }
                } else {
                    double[] zd = (double[]) z.array();
                    for (int i = 0; i < n; i++) {
                        zd[offsetZ + i * incrZ] = yd[offsetY + i * incrY] / xd[offsetX + i * incrX];
                    }
                }
            }
        }
    }

    @Override
    public TransformDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        return new RDivOpDataBufferAction(op,threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

}
