package org.nd4j.linalg.api.parallel.ops;

import lombok.AllArgsConstructor;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
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
                t1.fork();

                int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
                int offsetX2 = offsetX + nFirst * incrX;
                int offsetY2 = offsetY + nFirst * incrY;
                int offsetZ2 = offsetZ + nFirst * incrZ;
                BaseDataBufferAction t2 = getSubTask(threshold, nSecond, x, y, z, offsetX2, offsetY2, offsetZ2, incrX, incrY, incrZ);
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

        public BaseAccumulationDataBufferTask( Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
            this.op = op;
            this.threshold = threshold;
            this.outerTask = outerTask;
            INDArray tadX = x.tensorAlongDimension(tadIdx,tadDim);
            INDArray tadY = (y!=null ? y.tensorAlongDimension(tadIdx,tadDim) : null);
            this.x = x.data();
            this.y = (y != null ? y.data() : null);
            this.offsetX = tadX.offset();
            this.offsetY = (tadY != null ? tadY.offset() : 0);
            this.incrX = tadX.elementWiseStride();
            this.incrY = (tadY != null ? tadY.elementWiseStride() : 0);
            this.n = tadX.length();
        }

        @Override
        protected Double compute() {
            if (n > threshold) {
                //Split task
                int nFirst = n / 2;
                BaseAccumulationDataBufferTask t1 = getSubTask(threshold, nFirst, x, y, offsetX, offsetY, incrX, incrY, false);
                t1.fork();

                int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
                int offsetX2 = offsetX + nFirst * incrX;
                int offsetY2 = offsetY + nFirst * incrY;
                BaseAccumulationDataBufferTask t2 = getSubTask(threshold, nSecond, x, y, offsetX2, offsetY2, incrX, incrY, false);
                t2.fork();

                double first = t1.join();
                double second = t2.join();
                double preFinalResult = op.combineSubResults(first, second);
                if (outerTask) return op.getAndSetFinalResult(preFinalResult);
                else return preFinalResult;
            } else {
                return doTask();
            }
        }

        public abstract double doTask();

        public abstract BaseAccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y,
                                                                  int offsetX, int offsetY, int incrX, int incrY, boolean outerTask);
    }

    @AllArgsConstructor
    public static class TransformViaTensorDataBufferTask extends RecursiveAction {
        protected final TransformOp op;
        protected final int threshold;
        protected final INDArray x;
        protected final INDArray y;
        protected final INDArray z;

        @Override
        protected void compute() {
            //Break the transform op into tensors
            //Run transform on each tensor

            int tensorDim;
            if(y==null){
                if(x==z){
                    //x=Op(x)
                    tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
                } else {
                    //z=Op(x)
                    tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, z);
                }
            } else {
                if(x==z){
                    //x=Op(x,y)
                    tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y);
                } else {
                    //z=Op(x,y)
                    tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y, z);
                }
            }

            int nTensors = x.tensorssAlongDimension(tensorDim);
            if(nTensors == 1){
                new TransformOpDataBufferAction(op,0,tensorDim,threshold,x,y,z).invoke();
            } else {
                List<TransformOpDataBufferAction> blockList = new ArrayList<>(nTensors);
                if(x.rank() == 2) {
                    //Use fast tensor calculation for 2d
                    OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                    int n = tsx.getTensorLength();
                    int incrX = tsx.getElementWiseStride();
                    DataBuffer dx = x.data();
                    if(y==null){
                        if(x==z){
                            //x=Op(x)
                            for( int i=0; i<nTensors; i++){
                                int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                                TransformOpDataBufferAction task = new TransformOpDataBufferAction(op,threshold,n,dx,null,dx,offsetX,
                                        0,offsetX,incrX,0,incrX);
                                task.fork();
                                blockList.add(task);
                            }
                        } else {
                            //z=Op(x)
                            DataBuffer dz = z.data();
                            OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                            int incrZ = tsz.getElementWiseStride();
                            for( int i=0; i<nTensors; i++){
                                int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                                int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                                TransformOpDataBufferAction task = new TransformOpDataBufferAction(op,threshold,n,dx,null,dz,offsetX,
                                        0,offsetZ,incrX,0,incrZ);
                                task.fork();
                                blockList.add(task);
                            }
                        }
                    } else {
                        DataBuffer dy = y.data();
                        OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                        int incrY = tsy.elementWiseStride;
                        if(x==z){
                            //x=Op(x,y)
                            for( int i=0; i<nTensors; i++){
                                int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                                int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                                TransformOpDataBufferAction task = new TransformOpDataBufferAction(op,threshold,n,dx,dy,dx,offsetX,
                                        offsetY,offsetX,incrX,incrY,incrX);
                                task.fork();
                                blockList.add(task);
                            }
                        } else {
                            //z=Op(x,y)
                            DataBuffer dz = z.data();
                            OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                            int incrZ = tsz.getElementWiseStride();
                            for( int i=0; i<nTensors; i++){
                                int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                                int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                                int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                                TransformOpDataBufferAction task = new TransformOpDataBufferAction(op,threshold,n,dx,dy,dz,offsetX,
                                        offsetY,offsetZ,incrX,incrY,incrZ);
                                task.fork();
                                blockList.add(task);
                            }
                        }
                    }
                } else {
                    //Use general purpose tensor calculation for everything else
                    for (int i = 0; i < nTensors; i++) {
                        TransformOpDataBufferAction task = new TransformOpDataBufferAction(op, i, tensorDim, threshold, x, y, z);
                        task.fork();
                        blockList.add(task);
                    }
                }

                //Block until all are completed
                for(TransformOpDataBufferAction task : blockList){
                    task.join();
                }
            }
        }
    }

    @AllArgsConstructor
    public static class AccumulationViaTensorDataBufferTask extends RecursiveTask<Double> {
        protected final Accumulation op;
        protected final int threshold;
        protected final INDArray x;
        protected final INDArray y;

        @Override
        protected Double compute() {
            //Break the accumulation op into tensors
            //Run accumulation on each tensor
            //And combine the results

            int tensorDim;
            if(y==null) tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
            else tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x,y);

            int nTensors = x.tensorssAlongDimension(tensorDim);
            if(nTensors == 1){
                return new AccumulationOpDataBufferTask(op,0,tensorDim,threshold,x,y,true).invoke();
            } else {
                List<AccumulationOpDataBufferTask> blockList = new ArrayList<>(nTensors);

                if(x.rank()==2){
                    //Use fast tensor calculation for 2d
                    OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                    int n = tsx.getTensorLength();
                    int incrX = tsx.getElementWiseStride();
                    DataBuffer dx = x.data();
                    if(y==null){
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            AccumulationOpDataBufferTask task = new AccumulationOpDataBufferTask(op,threshold,n,dx,null,offsetX,0,incrX,0,false);
                            task.fork();
                            blockList.add(task);
                        }
                    } else {
                        DataBuffer dy = y.data();
                        OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                        int incrY = tsy.getElementWiseStride();
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            AccumulationOpDataBufferTask task = new AccumulationOpDataBufferTask(op,threshold,n,dx,dy,offsetX,offsetY,incrX,incrY,false);
                            task.fork();
                            blockList.add(task);
                        }
                    }
                } else {
                    //3+ dimensions
                    for( int i=0; i<nTensors; i++ ){
                        AccumulationOpDataBufferTask task = new AccumulationOpDataBufferTask(op,i,tensorDim,threshold,x,y,false);
                        task.fork();
                        blockList.add(task);
                    }
                }

                double accum = op.zeroDouble();
                for(AccumulationOpDataBufferTask task : blockList){
                    double subAccum = task.join();
                    accum = op.combineSubResults(accum,subAccum);
                }
                return op.getAndSetFinalResult(accum);
            }
        }
    }

    @AllArgsConstructor
    public static abstract class BaseIndexAccumulationDataBufferTask extends RecursiveTask<Pair<Double,Integer>> {
        protected final IndexAccumulation op;
        protected final int threshold;
        protected final int n;
        protected final DataBuffer x;
        protected final DataBuffer y;
        protected final int offsetX;    //Data buffer offset
        protected final int offsetY;
        protected final int incrX;
        protected final int incrY;
        protected final int elementOffset;  //Starting index of the first element
        protected final boolean outerTask;

        public BaseIndexAccumulationDataBufferTask( IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
            this.op = op;
            this.threshold = threshold;
            this.outerTask = outerTask;
            INDArray tadX = x.tensorAlongDimension(tadIdx,tadDim);
            INDArray tadY = (y!=null ? y.tensorAlongDimension(tadIdx,tadDim) : null);
            this.x = x.data();
            this.y = (y != null ? y.data() : null);
            this.offsetX = tadX.offset();
            this.offsetY = (tadY != null ? tadY.offset() : 0);
            this.incrX = tadX.elementWiseStride();
            this.incrY = (tadY != null ? tadY.elementWiseStride() : 0);
            this.n = tadX.length();

            this.elementOffset = tadIdx*tadX.length();  //First element of this tensor has index of elementOffset in original NDArray
        }

        @Override
        protected Pair<Double,Integer> compute() {
            if(n>threshold){
                //Split task
                int nFirst = n / 2;
                BaseIndexAccumulationDataBufferTask t1 = getSubTask(op,threshold,nFirst,x,y,offsetX,offsetY,incrX,incrY,elementOffset,false);
                t1.fork();

                int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
                int elementOffset2 = elementOffset + nFirst;
                int offsetX2 = offsetX + nFirst * incrX;
                int offsetY2 = offsetY + nFirst * incrY;
                BaseIndexAccumulationDataBufferTask t2 = getSubTask(op,threshold,nSecond,x,y,offsetX2,offsetY2,incrX,incrY,elementOffset2,false);
                t2.fork();

                Pair<Double,Integer> p1 = t1.join();
                Pair<Double,Integer> p2 = t2.join();

                Pair<Double,Integer> out = op.combineSubResults(p1,p2);
                if(outerTask) op.setFinalResult(out.getSecond());
                return out;
            } else {
                Pair<Double,Integer> out = doTask();
                if(outerTask) op.setFinalResult(out.getSecond());
                return out;
            }
        }

        public abstract Pair<Double,Integer> doTask();

        public abstract BaseIndexAccumulationDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                                                  int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask);
    }

    @AllArgsConstructor
    public static class IndexAccumulationViaTensorDataBufferTask extends RecursiveTask<Pair<Double,Integer>> {
        protected final IndexAccumulation op;
        protected final int threshold;
        protected final INDArray x;
        protected final INDArray y;

        @Override
        protected Pair<Double,Integer> compute() {
            //Break the index accumulation op into tensors
            //Run index accumulation on each tensor
            //And combine the results

            int tensorDim;
            if(y==null) tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
            else tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x,y);

            int nTensors = x.tensorssAlongDimension(tensorDim);
            if(nTensors == 1) {
                return new IndexAccumulationOpDataBufferTask(op, 0, tensorDim, threshold, x, y, true).invoke();
            } else {
                List<IndexAccumulationOpDataBufferTask> blockList = new ArrayList<>(nTensors);
                if(x.rank()==2){
                    //Use fast tensor calculation for 2d
                    OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                    int n = tsx.getTensorLength();
                    int incrX = tsx.getElementWiseStride();
                    DataBuffer dx = x.data();
                    if(y==null){
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int elementOffset = i*tsx.getTensorLength();
                            IndexAccumulationOpDataBufferTask task = new IndexAccumulationOpDataBufferTask(op,threshold,n,dx,null,offsetX,0,incrX,0,elementOffset,false);
                            task.fork();
                            blockList.add(task);
                        }
                    } else {
                        DataBuffer dy = y.data();
                        OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                        int incrY = tsy.getElementWiseStride();
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            int elementOffset = i*tsx.getTensorLength();
                            IndexAccumulationOpDataBufferTask task = new IndexAccumulationOpDataBufferTask(op,threshold,n,dx,dy,offsetX,offsetY,incrX,incrY,elementOffset,false);
                            task.fork();
                            blockList.add(task);
                        }
                    }
                } else {
                    //3+ dimensions
                    for( int i=0; i<nTensors; i++ ){
                        IndexAccumulationOpDataBufferTask task = new IndexAccumulationOpDataBufferTask(op,i,tensorDim,threshold,x,y,false);
                        task.fork();
                        blockList.add(task);
                    }
                }

                Pair<Double,Integer> accum = op.zeroPair();
                for(IndexAccumulationOpDataBufferTask task : blockList){
                    Pair<Double,Integer> subAccum = task.join();
                    accum = op.combineSubResults(accum,subAccum);
                }
                op.setFinalResult(accum.getSecond());
                return accum;
            }
        }
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

    public static class TransformOpDataBufferAction extends BaseDataBufferAction {
        private final TransformOp op;

        public TransformOpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            this.op = op;
        }

        public TransformOpDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
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
            return new TransformOpDataBufferAction(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }

    public static class ScalarOpDataBufferAction extends BaseDataBufferAction {
        private final ScalarOp op;

        public ScalarOpDataBufferAction(ScalarOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            super(threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            this.op = op;
        }

        public ScalarOpDataBufferAction(ScalarOp op, int tensorNum, int tensorDim, int threshold, INDArray x, INDArray z){
            super(tensorNum,tensorDim,threshold,x,null,z);
            this.op = op;
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
        public BaseDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new ScalarOpDataBufferAction(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        }
    }


    public static class AccumulationOpDataBufferTask extends BaseAccumulationDataBufferTask {
        public AccumulationOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                            int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
            super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
        }

        public AccumulationOpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
            super(op,tadIdx,tadDim,threshold,x,y,outerTask);
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
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
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
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
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
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
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
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
                }
            }
        }

        @Override
        public BaseAccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                         int incrX, int incrY, boolean outerTask) {
            return new AccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
        }
    }


    public static class IndexAccumulationOpDataBufferTask extends BaseIndexAccumulationDataBufferTask {
        public IndexAccumulationOpDataBufferTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                            int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask) {
            super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
        }

        public IndexAccumulationOpDataBufferTask(IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
            super(op,tadIdx,tadDim,threshold,x,y,outerTask);
        }

        @Override
        public Pair<Double,Integer> doTask() {
            if (y != null) {
                //Task: accum = update(accum,X,Y)
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    float accum = op.zeroFloat();
                    int idxAccum = -1;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xf[offsetX+i],yf[offsetY+i],i);
                            if(idxAccum==i) accum = op.op(xf[offsetX+i],yf[offsetY+i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xf[offsetX+i*incrX],yf[offsetY+i*incrY],i);
                            if(idxAccum==i) accum = op.op(xf[offsetX+i*incrX],yf[offsetY+i*incrY]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if(outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double)accum,finalIdx);
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    double accum = op.zeroDouble();
                    int idxAccum = -1;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xd[offsetX+i],yd[offsetY+i],i);
                            if(idxAccum==i) accum = op.op(xd[offsetX+i],yd[offsetY+i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xd[offsetX+i*incrX],yd[offsetY+i*incrY],i);
                            if(idxAccum==i) accum = op.op(xd[offsetX+i*incrX],yd[offsetY+i*incrY]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;
                    if(outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum,finalIdx);
                }
            } else {
                //Task: accum = update(accum,X)
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float accum = op.zeroFloat();
                    int idxAccum = -1;
                    if (incrX == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xf[offsetX+i],i);
                            if(idxAccum==i) accum = op.op(xf[offsetX+i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xf[offsetX+i*incrX],i);
                            if(idxAccum==i) accum = op.op(xf[offsetX+i*incrX]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;
                    if(outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double)accum,finalIdx);
                } else {
                    double[] xd = (double[]) x.array();
                    double accum = op.zeroDouble();
                    int idxAccum = -1;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xd[offsetX+i],i);
                            if(idxAccum==i) accum = op.op(xd[offsetX+i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum,idxAccum,xd[offsetX+i*incrX],i);
                            if(idxAccum==i) accum = op.op(xd[offsetX+i*incrX]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;
                    if(outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum,finalIdx);
                }
            }
        }

        @Override
        public BaseIndexAccumulationDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                         int incrX, int incrY, int elementOffset, boolean outerTask) {
            return new IndexAccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
        }
    }
}
