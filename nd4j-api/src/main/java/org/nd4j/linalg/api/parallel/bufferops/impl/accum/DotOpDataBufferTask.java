package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

public class DotOpDataBufferTask extends AccumulationDataBufferTask {

    public DotOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                               int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public DotOpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask) {
        super(op, tadIdx, tadDim, threshold, x, y, outerTask);
    }

    @Override
    public double doTask() {
        //Task: dotProduct(x,y)
        if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            double accum = Nd4j.getBlasWrapper().level1().dot(n, x, offsetX, incrX, y, offsetY, incrY);
            if(outerTask) return op.getAndSetFinalResult(accum);
            return accum;
        } else {
            ByteBuf nbbx = x.asNetty();
            ByteBuf nbby = y.asNetty();
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float accum = op.zeroFloat();
                int byteOffsetX = 4 * offsetX;
                int byteOffsetY = 4 * offsetY;
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < 4 * n; i += 4) {
                        accum = op.update(accum, nbbx.getFloat(byteOffsetX + i), nbby.getFloat(byteOffsetY + i));
                    }
                } else {
                    for (int i = 0; i < 4 * n; i += 4) {
                        accum = op.update(accum, nbbx.getFloat(byteOffsetX + i * incrX), nbby.getFloat(byteOffsetY + i * incrY));
                    }
                }
                if(outerTask) return op.getAndSetFinalResult(accum);
                return accum;
            } else {
                double accum = op.zeroDouble();
                int byteOffsetX = 8 * offsetX;
                int byteOffsetY = 8 * offsetY;
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < 8 * n; i += 8) {
                        accum = op.update(accum, nbbx.getDouble(byteOffsetX + i), nbby.getDouble(byteOffsetY + i));
                    }
                } else {
                    for (int i = 0; i < 8 * n; i += 8) {
                        accum = op.update(accum, nbbx.getDouble(byteOffsetX + i * incrX), nbby.getDouble(byteOffsetY + i * incrY));
                    }
                }
                if(outerTask) return op.getAndSetFinalResult(accum);
                return accum;
            }
        }
    }

    @Override
    public AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                 int incrX, int incrY, boolean outerTask) {
        return new DotOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }
}
