package org.nd4j.linalg.api.parallel.tasks.cpu.vector;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

public class CpuBroadcastOp extends BaseCPUAction {
    protected final BroadcastOp op;

    public CpuBroadcastOp(BroadcastOp op, int threshold) {
        super(op,threshold);
        this.op = op;
    }

    @Override
    public Void blockUntilComplete() {
        if(future==null){
            //invokeAsync() not called?
            invokeAsync();
        }

        try{
            future.get();
        }catch(Exception e){
            throw new RuntimeException(e);
        }

        if(subTasks!=null) {
            //i.e., for Callable/ExecutorService execution
            for (Task<Void> task : subTasks) {
                task.blockUntilComplete();
            }
        }

        return null;
    }

    @Override
    public Void call() {
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        int nVectorOps;
        if(x.rank() == 2 ){
            if(op.getDimension()[0] == 0) nVectorOps=x.columns();
            else nVectorOps = x.rows();
        } else {
            int[] shape = x.shape();
            nVectorOps = ArrayUtil.prod(ArrayUtil.removeIndex(shape, op.getDimension()));
        }

        subTasks = new ArrayList<>(nVectorOps);
        //Do TAD and create sub-tasks:
        int[] dimension = op.getDimension();
        if(x.size(dimension[0]) != y.length())
            throw new UnsupportedOperationException("Array length " + y.length() + " does not match x.shape("+dimension+")="+x.size(dimension[0]));

        if(x.rank() == 2) {
            OpExecutionerUtil.Tensor1DStats t1dx = OpExecutionerUtil.get1DTensorStats(x,dimension);
            if(y!=null) {
                int offsetY = y.offset();
                int ewsy = y.elementWiseStride();
                if (x == z) {
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        Task<Void> task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, offsetY, offsetX,
                                t1dx.getElementWiseStride(), ewsy, t1dx.getElementWiseStride());
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                } else {
                    OpExecutionerUtil.Tensor1DStats t1dz = OpExecutionerUtil.get1DTensorStats(z, dimension);
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        int offsetZ = t1dz.getFirstTensorOffset() + i * t1dz.getTensorStartSeparation();
                        Task<Void> task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, offsetY, offsetZ,
                                t1dx.getElementWiseStride(), ewsy, t1dz.getElementWiseStride());
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        Task<Void> task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, 0, offsetX,
                                t1dx.getElementWiseStride(), 0, t1dx.getElementWiseStride());
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                } else {
                    OpExecutionerUtil.Tensor1DStats t1dz = OpExecutionerUtil.get1DTensorStats(z, dimension);
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        int offsetZ = t1dz.getFirstTensorOffset() + i * t1dz.getTensorStartSeparation();
                        Task<Void> task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, 0, offsetZ,
                                t1dx.getElementWiseStride(), 0, t1dz.getElementWiseStride());
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            }
        } else {
            for( int i = 0; i < nVectorOps; i++ ) {
                Task<Void> task = new SingleVectorAction(threshold, i, dimension[0]);
                task.invokeAsync();
                subTasks.add(task);
            }
        }
        return null;
    }

    @Override
    protected void compute() {
        //Fork join
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        int nVectorOps;
        if(x.rank() == 2 ){
            if(op.getDimension()[0] == 0) nVectorOps=x.columns();
            else nVectorOps = x.rows();
        } else {
            int[] shape = x.shape();
            nVectorOps = ArrayUtil.prod(ArrayUtil.removeIndex(shape, op.getDimension()));
        }

        List<RecursiveAction> subTasks = new ArrayList<>(nVectorOps);
        //Do TAD and create sub-tasks:
        int[] dimension = op.getDimension();
        if(x.size(dimension[0]) != y.length())
            throw new UnsupportedOperationException("Vector length " + y.length() + " does not match x.shape(" + dimension[0] + ")= " + x.size(dimension[0]));

        if(x.rank() == 2) {
            OpExecutionerUtil.Tensor1DStats t1dx = OpExecutionerUtil.get1DTensorStats(x,dimension);
            if(y!=null) {
                int offsetY = y.offset();
                int ewsy = y.elementWiseStride();
                if (x == z) {
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        RecursiveAction task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, offsetY, offsetX,
                                t1dx.getElementWiseStride(), ewsy, t1dx.getElementWiseStride());
                        task.fork();
                        subTasks.add(task);
                    }
                } else {
                    OpExecutionerUtil.Tensor1DStats t1dz = OpExecutionerUtil.get1DTensorStats(z, dimension);
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        int offsetZ = t1dz.getFirstTensorOffset() + i * t1dz.getTensorStartSeparation();
                        RecursiveAction task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, offsetY, offsetZ,
                                t1dx.getElementWiseStride(), ewsy, t1dz.getElementWiseStride());
                        task.fork();
                        subTasks.add(task);
                    }
                }
            } else {
                if (x == z) {
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        RecursiveAction task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, 0, offsetX,
                                t1dx.getElementWiseStride(), 0, t1dx.getElementWiseStride());
                        task.fork();
                        subTasks.add(task);
                    }
                } else {
                    OpExecutionerUtil.Tensor1DStats t1dz = OpExecutionerUtil.get1DTensorStats(z, dimension);
                    for (int i = 0; i < nVectorOps; i++) {
                        int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                        int offsetZ = t1dz.getFirstTensorOffset() + i * t1dz.getTensorStartSeparation();
                        RecursiveAction task = new SingleVectorAction(threshold, t1dx.getTensorLength(), offsetX, 0, offsetZ,
                                t1dx.getElementWiseStride(), 0, t1dz.getElementWiseStride());
                        task.fork();
                        subTasks.add(task);
                    }
                }
            }
        } else {
            for( int i=0; i<nVectorOps; i++ ) {
                RecursiveAction task = new SingleVectorAction(threshold, i, dimension[0]);
                task.fork();
                subTasks.add(task);
            }
        }

        for(RecursiveAction a : subTasks ){
            a.join();
        }
    }

    /** helper class for doing a single vector op */
    private class SingleVectorAction extends BaseCPUAction {

        private SingleVectorAction(int threshold, int n, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }

        private SingleVectorAction(int threshold, int tadIdx, int tadDim) {
            super(op,threshold,tadIdx,tadDim);
        }

        @Override
        public void invokeAsync() {
            future = TaskExecutorProvider.getTaskExecutor().executeAsync(this);
        }

        @Override
        protected void compute() {
            if(doTensorFirst) doTensorFirst(op);

            //Fork join
            if(n>threshold) {
                int nFirst = n/2;
                RecursiveAction first = new SingleVectorAction(threshold, nFirst, offsetX, offsetY, offsetZ, incrX,
                        incrY, incrZ);
                first.fork();

                int nSecond = n - nFirst;
                int offsetX2 = offsetX + nFirst * incrX;
                int offsetY2 = offsetY + nFirst * incrY;
                int offsetZ2 = offsetZ + nFirst * incrZ;
                RecursiveAction second = new SingleVectorAction(threshold, nSecond, offsetX2, offsetY2, offsetZ2, incrX,
                        incrY, incrZ);
                second.fork();

                first.join();
                second.join();
            } else {
                execute();
            }
        }

        @Override
        public Void blockUntilComplete() {
            if(future != null){
                try{
                    future.get();
                }catch( Exception e ){
                    throw new RuntimeException(e);
                }
            } else {
                for( Task<Void> t : subTasks ){
                    t.blockUntilComplete();
                }
            }
            return null;
        }

        @Override
        public Void call() {
            //Callable
            if(n>threshold) {
                if(doTensorFirst) doTensorFirst(op);

                //Break into subtasks
                int nSubTasks = 1 + n / threshold;  //(round up)
                subTasks = new ArrayList<>(nSubTasks);
                //break into equal sized tasks:

                int taskSize = n / nSubTasks;
                int soFar = 0;
                for( int i=0; i<nSubTasks; i++ ){
                    int nInTask;
                    if(i==nSubTasks-1){
                        //All remaining tasks (due to integer division)
                        nInTask = n - soFar;
                    } else {
                        nInTask = taskSize;
                    }
                    int offsetXNew = offsetX + soFar*incrX;
                    int offsetYNew = offsetY + soFar*incrY;
                    int offsetZNew = offsetZ + soFar*incrZ;

                    SingleVectorAction task = new SingleVectorAction(threshold,nInTask,offsetXNew,offsetYNew,offsetZNew,
                            incrX,incrY,incrZ);
                    task.invokeAsync();
                    subTasks.add(task);

                    soFar += nInTask;
                }
            } else {
                execute();
            }
            return null;
        }

        @Override
        public void doTensorFirst(Op op){
            //Need to override BaseCPUAction.doTensorFirst(), as that does tensors for all of X, Y, Z
            //whereas here Y is already a vector -> only want to do tensors for X and Z

            INDArray x = op.x();
            INDArray y = op.y();
            INDArray z = op.z();
            INDArray tadx = x.tensorAlongDimension(tensorIdx, tensorDim);
            this.n = tadx.length();
            offsetX = tadx.offset();
            incrX = tadx.elementWiseStride();
            if (y == null) {
                offsetY = 0;
                incrY = 0;
            } else {
                offsetY = y.offset();
                incrY = y.elementWiseStride();
            }

            if (z == null) {
                offsetZ = 0;
                incrZ = 0;
            } else if (z == x) {
                offsetZ = offsetX;
                incrZ = incrX;
            } else {
                INDArray tadz = z.tensorAlongDimension(tensorIdx, tensorDim);
                offsetZ = tadz.offset();
                incrZ = tadz.elementWiseStride();
            }
        }

        private Void execute(){
            DataBuffer x = op.x().data();
            DataBuffer y = op.y().data();
            DataBuffer z = op.z().data();

            if(x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                if(x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
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
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
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
                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbby = y.asNetty();
                ByteBuf nbbz = z.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetY = 4 * offsetY;
                    int byteOffsetZ = 4 * offsetZ;
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < 4*n; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i), nbby.getFloat(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < 4*n; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                } else {
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    int byteOffsetZ = 8 * offsetZ;
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < 8*n; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < 8*n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i), nbby.getDouble(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < 8*n; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < 8*n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                }
            }
            return null;
        }
    }
}
