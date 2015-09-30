package org.nd4j.linalg.api.parallel.ops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

/**
 * @author Alex Black
 */
public class BufferOps {

    @AllArgsConstructor
    public static abstract class BaseDataBufferAction extends RecursiveAction {
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
        public BaseDataBufferAction(int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
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
                BaseDataBufferAction t1 = getSubTask(threshold, nFirst, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);

                int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
                int offsetX2 = offsetX + nFirst * incrX;
                int offsetY2 = offsetY + nFirst * incrY;
                int offsetZ2 = offsetZ + nFirst * incrZ;
                BaseDataBufferAction t2 = getSubTask(threshold, nSecond, x, y, z, offsetX2, offsetY2, offsetZ2, incrX, incrY, incrZ);

                t1.fork();
                t2.fork();
                t1.join();
                t2.join();
            } else {
                doTask();
            }
        }

        public abstract void doTask();

        public abstract BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z,
                                                        int offsetX, int offsetY, int offsetZ,
                                                        int incrX, int incrY, int incrZ);
    }

    @AllArgsConstructor
    public static abstract class BaseAccumulationDataBufferTask extends RecursiveTask<Double> {
        protected final Accumulation op;
        protected final int threshold;
        protected final int n;
        protected final DataBuffer x;
        protected final DataBuffer y;
        protected final int offsetX;
        protected final int offsetY;
        protected final int incrX;
        protected final int incrY;
        protected final boolean outerTask;

        @Override
        protected Double compute() {
            if (n > threshold) {
                //Split task
                int nFirst = n / 2;
                BaseAccumulationDataBufferTask t1 = getSubTask(threshold, nFirst, x, y, offsetX, offsetY, incrX, incrY, false);

                int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
                int offsetX2 = offsetX + nFirst * incrX;
                int offsetY2 = offsetY + nFirst * incrY;
                BaseAccumulationDataBufferTask t2 = getSubTask(threshold, nSecond, x, y, offsetX2, offsetY2, incrX, incrY, false);

                t1.fork();
                t2.fork();
                double first = t1.join();
                double second = t2.join();
                double preFinalResult = op.combineSubResults(first, second);
                if (outerTask) return op.getFinalResult(preFinalResult);
                else return preFinalResult;
            } else {
                return doTask();
            }
        }

        public abstract double doTask();

        public abstract BaseAccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y,
                                                                  int offsetX, int offsetY, int incrX, int incrY, boolean outerTask);
    }


    public static class AddOpDataBufferAction extends BaseDataBufferAction {
        public AddOpDataBufferAction(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }

        public AddOpDataBufferAction(int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
            super(tadIdx, tadDim, threshold, x, y, z);
        }

        @Override
        public void doTask() {
            //Task: Z = X+Y
            if (x == z) {
                //can use axpy
                Nd4j.getBlasWrapper().level1().axpy(n, 1.0, y, offsetY, incrY, z, offsetZ, incrZ);
            } else {
                //use loop
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    float[] zf = (float[]) z.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i] = xf[offsetX + i] + yf[offsetY + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] + yf[offsetY + i * incrY];
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    double[] zd = (double[]) z.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = xd[offsetX + i] + yd[offsetY + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] + yd[offsetY + i * incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new AddOpDataBufferAction(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }

    public static class SubOpDataBufferAction extends BaseDataBufferAction {
        public SubOpDataBufferAction(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }

        public SubOpDataBufferAction(int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
            super(tadIdx, tadDim, threshold, x, y, z);
        }

        @Override
        public void doTask() {
            //Task: Z = X+Y
            if (x == z) {
                //can use axpy
                Nd4j.getBlasWrapper().level1().axpy(n, -1.0, y, offsetY, incrY, z, offsetZ, incrZ);
            } else {
                //use loop
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    float[] zf = (float[]) z.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i] = xf[offsetX + i] - yf[offsetY + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] - yf[offsetY + i * incrY];
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    double[] zd = (double[]) z.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = xd[offsetX + i] - yd[offsetY + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] - yd[offsetY + i * incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new SubOpDataBufferAction(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }


    public static class MulOpDataBufferAction extends BaseDataBufferAction {
        public MulOpDataBufferAction(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }

        public MulOpDataBufferAction(int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
            super(tadIdx, tadDim, threshold, x, y, z);
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
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new MulOpDataBufferAction(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }

    public static class DivOpDataBufferAction extends BaseDataBufferAction {
        public DivOpDataBufferAction(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }

        public DivOpDataBufferAction(int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
            super(tadIdx, tadDim, threshold, x, y, z);
        }

        @Override
        public void doTask() {
            //Task: Z = X/Y

            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float[] xf = (float[]) x.array();
                float[] yf = (float[]) y.array();
                if (incrX == 1 && incrY == 1 && incrZ == 1) {
                    if (x == z) {
                        for (int i = 0; i < n; i++) {
                            xf[offsetX + i] /= yf[offsetY + i];
                        }
                    } else {
                        float[] zf = (float[]) z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i] = xf[offsetX + i] / yf[offsetY + i];
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = 0; i < n; i++) {
                            xf[offsetX + i * incrX] /= yf[offsetY + i * incrY];
                        }
                    } else {
                        float[] zf = (float[]) z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] / yf[offsetY + i * incrY];
                        }
                    }
                }
            } else {
                double[] xd = (double[]) x.array();
                double[] yd = (double[]) y.array();
                if (incrX == 1 && incrY == 1 && incrZ == 1) {
                    if (x == z) {
                        for (int i = 0; i < n; i++) {
                            xd[offsetX + i] /= yd[offsetY + i];
                        }
                    } else {
                        double[] zd = (double[]) z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = xd[offsetX + i] / yd[offsetY + i];
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = 0; i < n; i++) {
                            xd[offsetX + i * incrX] /= yd[offsetY + i * incrY];
                        }
                    } else {
                        double[] zd = (double[]) z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] / yd[offsetY + i * incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new DivOpDataBufferAction(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }

    public static class CopyOpDataBufferAction extends BaseDataBufferAction {
        public CopyOpDataBufferAction(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }

        public CopyOpDataBufferAction(int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
            super(tadIdx, tadDim, threshold, x, y, z);
        }

        @Override
        public void doTask() {
            //Task: Z = X
            if (x == z) return;    //No op
            Nd4j.getBlasWrapper().level1().copy(n, x, offsetX, incrX, z, offsetZ, incrZ);
        }

        @Override
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new CopyOpDataBufferAction(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }

    public static class OpDataBufferAction extends BaseDataBufferAction {
        private final Op op;

        public OpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            this.op = op;
        }

        public OpDataBufferAction(ScalarOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            this.op = op;
        }

        public OpDataBufferAction(Op op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
            super(tadIdx, tadDim, threshold, x, y, z);
            this.op = op;
        }

        @Override
        public void doTask() {
            if (y != null) {
                //Task: Z = Op(X,Y)
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i;
                                xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = 0; i < n; i++) {

                                zf[offsetZ + i] = op.op(xf[offsetX + i], yf[offsetY + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i * incrX;
                                xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i * incrY]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                            }
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i;
                                xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zd[offsetZ + i] = op.op(xd[offsetX + i], yd[offsetY + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i * incrX;
                                xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i * incrY]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX], yd[offsetY + i * incrY]);
                            }
                        }
                    }
                }
            } else {
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
        }

        @Override
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            if (op instanceof TransformOp)
                return new OpDataBufferAction((TransformOp) op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            else
                return new OpDataBufferAction((ScalarOp) op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }


    public static class AccumulationOpDataBufferTask extends BaseAccumulationDataBufferTask {
        public AccumulationOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                            int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
            super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
        }

        @Override
        public double doTask() {
            if (y != null) {
                //Task: accum = update(accum,X,Y)
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    float accum = op.zeroFloat();
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xf[offsetX + i], yf[offsetY + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                        }
                    }
                    return accum;
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    double accum = op.zeroDouble();
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xd[offsetX + i], yd[offsetY + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xd[offsetX + i * incrX], yd[offsetY + i * incrY]);
                        }
                    }
                    return accum;
                }
            } else {
                //Task: accum = update(accum,X)
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float accum = op.zeroFloat();
                    if (incrX == 1) {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xf[offsetX + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xf[offsetX + i * incrX]);
                        }
                    }
                    return accum;
                } else {
                    double[] xd = (double[]) x.array();
                    double accum = op.zeroDouble();
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xd[offsetX + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            accum = op.update(accum, xd[offsetX + i * incrX]);
                        }
                    }
                    return accum;
                }
            }
        }

        @Override
        public BaseAccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                         int incrX, int incrY, boolean outerTask) {
            return new AccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
        }
    }

}
