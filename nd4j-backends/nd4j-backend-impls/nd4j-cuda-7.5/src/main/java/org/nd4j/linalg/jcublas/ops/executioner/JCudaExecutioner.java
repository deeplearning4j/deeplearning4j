/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas.ops.executioner;


import jcuda.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.CopyOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.DefaultPointerConverter;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.PointerConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * If requested Op doesn't exist within GPU context, DefaultOpExecutioner will be used, with arrays/buffers updated after that.
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public class JCudaExecutioner extends DefaultOpExecutioner {

    private static final Allocator allocator = AtomicAllocator.getInstance();
    private static NativeOps nativeOps = new NativeOps();
    private static Logger log = LoggerFactory.getLogger(JCudaExecutioner.class);
    public JCudaExecutioner() {
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};




        if(dimension[0] == Integer.MAX_VALUE|| dimension[0] == 0) {
            if(op.x() instanceof IComplexNDArray) {
                return Nd4j.scalar(execAndReturn(op).getFinalResultComplex());
            }
            return Nd4j.scalar(execAndReturn(op).getFinalResult().doubleValue());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).getFinalResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            //nothing to reduce
            if(ArrayUtil.prod(retShape) == op.x().length()) {
                return op.x();
            }

            invoke(op,dimension);
            if(op.z() == null)
                throw new IllegalStateException("No result set");
            return op.z();
        }


    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};



        if(dimension[0] == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).getFinalResult());
            return Nd4j.scalar(execAndReturn(op).getFinalResult());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).getFinalResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            //nothing to reduce
            if(ArrayUtil.prod(retShape) == op.x().length())
                return op.x();

            invoke(op,dimension);
            return op.z();
        }


    }


    @Override
    public INDArray execAndReturn(TransformOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }



    @Override
    public INDArray execAndReturn(ScalarOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }

    @Override
    public Op exec(Op op, int... dimension) {
        return super.exec(op, dimension);
    }


    @Override
    public Op exec(Op op) {
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpreted as vectors)
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA  || op instanceof CopyOp) {
            try {
                // we dont' care about op.Z sync state, since it'll be overwritten
                if (op.x() != null)
                    allocator.synchronizeHostData(op.x());
                if (op.y() != null)
                    allocator.synchronizeHostData(op.y());

                super.exec(op);
                return null;
            } finally {
                // we notify allocator that op.Z was modified on host side
                if (op.z() != null)
                    allocator.tickHostWrite(op.z());
            }
        }

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc,null);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc);
        } else if(op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            invoke(broadcastOp);
        }
        else if(op instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) op;
            invoke(indexAccumulation,null);
        }
        return op;
    }



    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op);
        return op.z();
    }







    private CudaContext invoke(BroadcastOp op) {
        long x = AtomicAllocator.getInstance().getDevicePointer(op.x()).getNativePointer();
        long xShapeInfo = AddressRetriever.retrieveDeviceAddress(op.x().shapeInfoDataBuffer());
        long[] xShapeInfoHostPointer = new long[]{
                AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()),
                AtomicAllocator.getInstance().getCudaContext().
                        getOldStream().getNativePointer()};

        long y = AtomicAllocator.getInstance().getDevicePointer(op.y()).getNativePointer();
        long yShapeInfo = AddressRetriever.retrieveDeviceAddress(op.y().shapeInfoDataBuffer());

        long z = AtomicAllocator.getInstance().getDevicePointer(op.z()).getNativePointer();
        long zShapeInfo = AddressRetriever.retrieveDeviceAddress(op.z().shapeInfoDataBuffer());
        long dimensionPointer = AddressRetriever.retrieveDeviceAddress(Nd4j.createBuffer(op.getDimension()));

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execBroadcastDouble(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    y,
                    yShapeInfo,
                    z,
                    zShapeInfo,
                    dimensionPointer,
                    op.getDimension().length);
        }
        else {
            nativeOps.execBroadcastFloat(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    y,
                    yShapeInfo,
                    z,
                    zShapeInfo,
                    dimensionPointer,
                    op.getDimension().length);

        }

        if (op.x() != null)
            allocator.tackDevice(op.x());
        if (op.y() != null)
            allocator.tackDevice(op.y());
        if (op.z() != null)
            allocator.tackDevice(op.z());

        // we notify allocator that op.Z was modified on device side
        if (op.z() != null)
            allocator.tickDeviceWrite(op.z());
        return null;
    }



    private CudaContext invoke(IndexAccumulation op,int[] dimension)  {
        long x = AtomicAllocator.getInstance().getDevicePointer(op.x()).getNativePointer();
        long xShapeInfo = AddressRetriever.retrieveDeviceAddress(op.x().shapeInfoDataBuffer());
        long extraArgs = op.extraArgs() != null ? AddressRetriever.retrieveDeviceAddress(op.extraArgsDataBuff()) : 0;

        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()),
                AtomicAllocator.getInstance().
                        getCudaContext().
                        getOldStream().getNativePointer()};
        if(op.z().isScalar() || dimension == null || dimension[0] == Integer.MAX_VALUE) {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                double result = nativeOps.execIndexReduceScalarDouble(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        extraArgs);
                op.setFinalResult((int) result);
            }
            else {
                float result = nativeOps.execIndexReduceScalarFloat(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        extraArgs);
                op.setFinalResult((int) result);
            }
        }
        else {
            if (dimension == null) dimension = new int[] {0};
            long z = AtomicAllocator.getInstance().getDevicePointer(op.z()).getNativePointer();
            long zShapeInfo = AddressRetriever.retrieveDeviceAddress(op.z().shapeInfoDataBuffer());
            long dimensionPointer = AddressRetriever.retrieveDeviceAddress(Nd4j.createBuffer(dimension));
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execIndexReduceDouble(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        extraArgs,
                        z,
                        zShapeInfo,
                        dimensionPointer,
                        dimension.length);
            }
            else {
                nativeOps.execIndexReduceFloat(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        extraArgs,
                        z,
                        zShapeInfo,
                        dimensionPointer,
                        dimension.length);
            }
        }

        if (op.x() != null)
            allocator.tackDevice(op.x());
        if (op.y() != null)
            allocator.tackDevice(op.y());
        if (op.z() != null)
            allocator.tackDevice(op.z());

        // we notify allocator that op.Z was modified on device side
         //if (op.x() != null) allocator.tickDeviceWrite(op.x());

        return null;

    }


    private CudaContext invoke(Accumulation op, int[] dimension) {
        // dimension is ALWAYS null here.
        if (dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};
        CudaContext ctx = null;
        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), AtomicAllocator.getInstance().getCudaContext().getOldStream().getNativePointer()};
        long x = AtomicAllocator.getInstance().getDevicePointer(op.x()).getNativePointer();
        long xShapeInfo = AddressRetriever.retrieveDeviceAddress(op.x().shapeInfoDataBuffer());
        long extraArgs = op.extraArgs() != null ? AddressRetriever.retrieveDeviceAddress(op.extraArgsDataBuff()) : 0;

        if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            if(op instanceof Variance) {
                double result = nativeOps.execSummaryStatsScalarDouble(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x
                        ,xShapeInfo,extraArgs, true);
                op.setFinalResult(result);
            } else if (op.y() != null) {
                long y = AtomicAllocator.getInstance().getDevicePointer(op.y()).getNativePointer();
                long yShapeInfo = AddressRetriever.retrieveDeviceAddress(op.y().shapeInfoDataBuffer());
                double result = nativeOps.execReduce3ScalarDouble(xShapeInfoHostPointer,
                        op.opNum()
                        , x,
                        xShapeInfo,
                        extraArgs,
                        y,
                        yShapeInfo);
                op.setFinalResult(result);
            } else {
                double result = nativeOps.execReduceScalarDouble(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        extraArgs);
                op.setFinalResult(result);
            }
        } else {
            if(op instanceof Variance) {
                float result = nativeOps.execSummaryStatsScalarFloat(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x
                        ,xShapeInfo,extraArgs, true);
                op.setFinalResult(result);
            } else if (op.y() != null) {
                long y = AtomicAllocator.getInstance().getDevicePointer(op.y()).getNativePointer();
                long yShapeInfo = AddressRetriever.retrieveDeviceAddress(op.y().shapeInfoDataBuffer());
                float result = nativeOps.execReduce3ScalarFloat(xShapeInfoHostPointer,
                        op.opNum()
                        , x,
                        xShapeInfo,
                        extraArgs,
                        y,
                        yShapeInfo);
                op.setFinalResult(result);
            } else {
                float result = nativeOps.execReduceScalarFloat(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        extraArgs);
                op.setFinalResult(result);
            }
        }


        if (op.x() != null)
            allocator.tackDevice(op.x());
        if (op.y() != null)
            allocator.tackDevice(op.y());
        if (op.z() != null)
            allocator.tackDevice(op.z());

        // we notify allocator that op.Z was modified on device side
        if (op.z() != null)
            allocator.tickDeviceWrite(op.z());

        return ctx;
    }


    private CudaContext invoke(ScalarOp op) {
        long x = AtomicAllocator.getInstance().getDevicePointer(op.x()).getNativePointer();
        long xShapeInfo = AddressRetriever.retrieveDeviceAddress(op.x().shapeInfoDataBuffer());
        long extraArgs = op.extraArgs() != null ? AddressRetriever.retrieveDeviceAddress(op.extraArgsDataBuff()) : 0;

        long z = AtomicAllocator.getInstance().getDevicePointer(op.z()).getNativePointer();
        long zShapeInfo = AddressRetriever.retrieveDeviceAddress(op.z().shapeInfoDataBuffer());
        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), AtomicAllocator.getInstance().getCudaContext().getOldStream().getNativePointer() };

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execScalarDouble(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    z,
                    zShapeInfo,
                    op.scalar().doubleValue(),
                    extraArgs);
        }
        else {
            nativeOps.execScalarFloat(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    z,
                    zShapeInfo,
                    op.scalar().floatValue(),
                    extraArgs);
        }

        if (op.x() != null)
            allocator.tackDevice(op.x());
        if (op.y() != null)
            allocator.tackDevice(op.y());
        if (op.z() != null)
            allocator.tackDevice(op.z());

        // we notify allocator that op.Z was modified on device side
        if (op.z() != null)
            allocator.tickDeviceWrite(op.z());
        return  null;
    }

    private CudaContext invoke(TransformOp op) {
        log.info("OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");
        long x = AtomicAllocator.getInstance().getDevicePointer(op.x()).getNativePointer();
        long xShapeInfo = AddressRetriever.retrieveDeviceAddress(op.x().shapeInfoDataBuffer());
        long extraArgs = op.extraArgs() != null ? AddressRetriever.retrieveDeviceAddress(op.extraArgsDataBuff()) : 0;

        long z = AtomicAllocator.getInstance().getDevicePointer(op.z()).getNativePointer();
        long zShapeInfo = AddressRetriever.retrieveDeviceAddress(op.z().shapeInfoDataBuffer());
        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), AtomicAllocator.getInstance().getCudaContext().getOldStream().getNativePointer()};


        if(op.y() != null) {
            long y = AtomicAllocator.getInstance().getDevicePointer(op.y()).getNativePointer();
            long yShapeInfo = AddressRetriever.retrieveDeviceAddress(op.y().shapeInfoDataBuffer());

            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execPairwiseTransformDouble(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        y,
                        yShapeInfo,
                        z,
                        zShapeInfo,
                        extraArgs);
            } else {
                nativeOps.execPairwiseTransformFloat(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        y,
                        yShapeInfo,
                        z,
                        zShapeInfo,
                        extraArgs);
            }
        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execTransformDouble(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        z,
                        zShapeInfo,
                        extraArgs);
            } else {
                nativeOps.execTransformFloat(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        x,
                        xShapeInfo,
                        z,
                        zShapeInfo,
                        extraArgs);
            }
        }
        if (op.x() != null)
            allocator.tackDevice(op.x());
        if (op.y() != null)
            allocator.tackDevice(op.y());
        if (op.z() != null)
            allocator.tackDevice(op.z());

        // we notify allocator that op.Z was modified on device side
        if (op.z() != null)
            allocator.tickDeviceWrite(op.z());

        return null;
    }
}


