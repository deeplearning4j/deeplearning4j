package org.nd4j.linalg.api.parallel.bufferops.impl.indexaccum;

import io.netty.buffer.ByteBuf;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.parallel.bufferops.IndexAccumulationDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

public class IAMaxOpDataBufferTask extends IndexAccumulationDataBufferTask {

    public IAMaxOpDataBufferTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                 int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }

    public IAMaxOpDataBufferTask(IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask) {
        super(op, tadIdx, tadDim, threshold, x, y, outerTask);
    }

    @Override
    public Pair<Double, Integer> doTask() {
        if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            //Can use BLAS iamax when heap allocated, but not always when direct allocated
            int idxAccum = Nd4j.getBlasWrapper().level1().iamax(n, x, offsetX, incrX);
            double value = x.getDouble(offsetX + idxAccum * incrX);
            idxAccum += this.elementOffset; //idxAccum gives relative to start of segment, not relative to start of NDArray
            if (outerTask) op.setFinalResult(idxAccum);
            return new Pair<>(value, idxAccum);
        } else {
            ByteBuf nbbx = x.asNetty();
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float accum = op.zeroFloat();
                int accumIdx = -1;
                int byteOffsetX = 4 * offsetX;
                if (incrX == 1) {
                    int j = 0;
                    for (int i = 0; i < 4 * n; i += 4, j++) {
                        float fx = nbbx.getFloat(byteOffsetX + i);
                        accumIdx = op.update(accum, accumIdx, fx, j);
                        if (accumIdx == j) accum = op.op(fx);
                    }
                } else {
                    int j = 0;
                    for (int i = 0; i < 4 * n; i += 4, j++) {
                        float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                        accumIdx = op.update(accum, accumIdx, fx, j);
                        if (accumIdx == j) accum = op.op(fx);
                    }
                }
                accumIdx += this.elementOffset; //idxAccum gives relative to start of segment, not relative to start of NDArray
                if (outerTask) op.setFinalResult(accumIdx);
                return new Pair<>((double) accum, accumIdx);
            } else {
                double accum = op.zeroDouble();
                int accumIdx = -1;
                int byteOffsetX = 8 * offsetX;
                if (incrX == 1) {
                    int j = 0;
                    for (int i = 0; i < 8 * n; i += 8, j++) {
                        double dx = nbbx.getDouble(byteOffsetX + i);
                        accumIdx = op.update(accum, accumIdx, dx, j);
                        if (accumIdx == j) accum = op.op(dx);
                    }
                } else {
                    int j = 0;
                    for (int i = 0; i < 8 * n; i += 8, j++) {
                        double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                        accumIdx = op.update(accum, accumIdx, dx, j);
                        if (accumIdx == j) accum = op.op(dx);
                    }
                }
                accumIdx += this.elementOffset; //idxAccum gives relative to start of segment, not relative to start of NDArray
                if (outerTask) op.setFinalResult(accumIdx);
                return new Pair<>(accum, accumIdx);
            }
        }
    }

    @Override
    public IndexAccumulationDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask) {
        return new IAMaxOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }
}
