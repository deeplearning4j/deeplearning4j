package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import io.netty.buffer.ByteBuf;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

public class Norm1OpDataBufferTask extends AccumulationDataBufferTask {

    public Norm1OpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                 int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public Norm1OpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask) {
        super(op, tadIdx, tadDim, threshold, x, y, outerTask);
    }

    @Override
    public double doTask() {
        //Task: sum_i |x_i|
        if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            double accum = Nd4j.getBlasWrapper().level1().asum(n, x, offsetX, incrX);
            if(outerTask) return op.getAndSetFinalResult(accum);
            return accum;
        } else {
            ByteBuf nbbx = x.asNetty();
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float accum = op.zeroFloat();
                int byteOffsetX = 4 * offsetX;
                if (incrX == 1) {
                    for (int i = 0; i < 4 * n; i += 4) {
                        accum = op.update(accum, FastMath.abs(nbbx.getFloat(byteOffsetX + i)));
                    }
                } else {
                    for (int i = 0; i < 4 * n; i += 4) {
                        accum = op.update(accum, FastMath.abs(nbbx.getFloat(byteOffsetX + i * incrX)));
                    }
                }
                if(outerTask) return op.getAndSetFinalResult(accum);
                return accum;
            } else {
                double accum = op.zeroDouble();
                int byteOffsetX = 8 * offsetX;
                if (incrX == 1) {
                    for (int i = 0; i < 8 * n; i += 8) {
                        accum = op.update(accum, FastMath.abs(nbbx.getDouble(byteOffsetX + i)));
                    }
                } else {
                    for (int i = 0; i < 8 * n; i += 8) {
                        accum = op.update(accum, FastMath.abs(nbbx.getDouble(byteOffsetX + i * incrX)));
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
        return new Norm1OpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }
}
