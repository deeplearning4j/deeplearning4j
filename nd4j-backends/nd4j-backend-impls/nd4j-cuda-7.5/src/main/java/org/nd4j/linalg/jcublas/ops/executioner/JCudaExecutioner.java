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



import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.CopyOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
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

    private static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    private static final Allocator allocator = AtomicAllocator.getInstance();
    private static Logger log = LoggerFactory.getLogger(JCudaExecutioner.class);
    public JCudaExecutioner() {

    }

    public NativeOps getNativeOps() {
        return nativeOps;
    }

    @Override
    protected void doBroadcastOp(BroadcastOp op) {
        exec(op);
    }

    @Override
    public INDArray exec(BroadcastOp op,int...dimension) {
        Arrays.sort(dimension);
        //log.info("B OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "], dimension: {}", Arrays.toString(dimension));

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());
        
        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
        long z = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long[] xShapeInfoHostPointer = new long[]{ AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar()};
        long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context).address();

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execBroadcastDouble(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    y,
                    AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address(),
                    z,
                    AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                    dimensionPointer, dimension.length);
        }
        else {
            nativeOps.execBroadcastFloat(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    y,
                    AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address(),
                    z,
                    AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                    dimensionPointer, dimension.length);
        }

        allocator.registerAction(context, op.z(), op.x(), op.y());

        return op.z();
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        Arrays.sort(dimension);

//        log.info("A2 OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");
//        log.info("op.x shape: " + Arrays.toString(op.x().shape()));
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};


        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[]{1, retShape[0]};
            else
                retShape = new int[]{retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[]{1, 1};
        }

        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.noOp();

        INDArray ret = null;
        if (op.zeroDouble() > -0.01f && op.zeroDouble() < 0.01f) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }
        op.setZ(ret);

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());


        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long[] xShapeInfoHostPointer = new long[]{ AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar()};
        long extraArgs = op.extraArgs() != null && op instanceof Variance ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;
        //long extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;
        long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context).address();

       // log.info("Extras: {}",op.extraArgsDataBuff());
        /*
        log.info("xShapeInfoHostPointer: " + Arrays.toString(xShapeInfoHostPointer));
        log.info("X: " + x);
        log.info("xShapeInfo: " + xShapeInfo);
*/
        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            if(op instanceof Variance) {
                if(ret.isScalar()) {
                    allocator.tickHostWrite(ret);

                    ret.putScalar(0, nativeOps.execSummaryStatsScalarDouble(xShapeInfoHostPointer, op.opNum(), x, xShapeInfo, extraArgs, true));

                    op.setFinalResult(ret.getDouble(0));
                } else {
                    nativeOps.execSummaryStatsDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.z(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                            dimensionPointer,
                            dimension.length,
                            ((Variance) op).isBiasCorrected()
                    );

                    allocator.registerAction(context, op.z(), op.x(), op.y());
                }
            } else if (op.y() != null) {
                if (ret.isScalar()) {
                    allocator.tickHostWrite(ret);

                    ret.putScalar(0, nativeOps.execReduce3ScalarDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.y(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address()
                    ));

                    op.setFinalResult(ret.getDouble(0));
                } else {
                    nativeOps.execReduce3Double(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.y(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                            dimensionPointer,
                            dimension.length
                    );

                    allocator.registerAction(context, op.z(), op.x(), op.y());
                }
            } else {
                if (ret.isScalar()) {
                    allocator.tickHostWrite(ret);

                    ret.putScalar(0, nativeOps.execReduceScalarDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs
                    ));

                    op.setFinalResult(ret.getDouble(0));
                } else {
                    nativeOps.execReduceDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.z(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                            dimensionPointer,
                            dimension.length
                    );

                    allocator.registerAction(context, op.z(), op.x(), op.y());
                }
            }
        } else {
            if(op instanceof Variance) {
                if(ret.isScalar()) {
                    allocator.tickHostWrite(ret);

                    ret.putScalar(0, nativeOps.execSummaryStatsScalarFloat(xShapeInfoHostPointer, op.opNum(), x, xShapeInfo, extraArgs, true));

                    op.setFinalResult(ret.getFloat(0));
                } else {
                    nativeOps.execSummaryStatsFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.z(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                            dimensionPointer,
                            dimension.length,
                            ((Variance) op).isBiasCorrected()
                    );

                    allocator.registerAction(context, op.z(), op.x(), op.y());
                }
            } else if (op.y() != null) {
                if (ret.isScalar()) {
                    allocator.tickHostWrite(ret);

                    ret.putScalar(0, nativeOps.execReduce3ScalarFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.y(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address()
                    ));

                    op.setFinalResult(ret.getFloat(0));
                } else {
                    nativeOps.execReduce3Float(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.y(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                            dimensionPointer,
                            dimension.length
                    );

                    allocator.registerAction(context, op.z(), op.x(), op.y());
                }
            } else {
                if (ret.isScalar()) {
                    allocator.tickHostWrite(ret);

                    ret.putScalar(0, nativeOps.execReduceScalarFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs
                    ));

                    op.setFinalResult(ret.getFloat(0));
                } else {
                    nativeOps.execReduceFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            AtomicAllocator.getInstance().getPointer(op.z(), context).address(),
                            AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address(),
                            dimensionPointer,
                            dimension.length
                    );

                    allocator.registerAction(context, op.z(), op.x(), op.y());
                }
            }
        }


        return ret;
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        Arrays.sort(dimension);

        //log.info("OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");


        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};



        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.x();


        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[]{1, retShape[0]};
            else
                retShape = new int[]{retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[]{1, 1};
        }

        INDArray ret = null;
        if (op.zeroDouble() > -0.01f && op.zeroDouble() < 0.01f) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }

        op.setZ(ret);
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();

        long z = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
        long zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address();
        long[] xShapeInfoHostPointer = new long[]{ AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar()};
        long extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;
        long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context).address();

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execIndexReduceDouble(
                    xShapeInfoHostPointer,
                    op.opNum(),
                    x,
                    xShapeInfo,
                    extraArgs,
                    z,
                    zShapeInfo,
                    dimensionPointer, dimension.length);

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
                    dimensionPointer, dimension.length);

        }

        allocator.registerAction(context, op.z(), op.x(), op.y());

        return op.z();
    }


    @Override
    public Op exec(Op op, int... dimension) {
        Arrays.sort(dimension);
        return super.exec(op, dimension);
    }


    @Override
    public Op exec(Op op) {
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpreted as vectors)
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA  || op instanceof CopyOp) {
                // we dont' care about op.Z sync state, since it'll be overwritten
                if (op.x() != null)
                    allocator.synchronizeHostData(op.x());
                if (op.y() != null)
                    allocator.synchronizeHostData(op.y());

                super.exec(op);

                if (op.z() != null)
                    allocator.tickHostWrite(op.z());
                return null;
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
        //log.info("OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");
        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long[] xShapeInfoHostPointer = new long[]{ AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar()};

        long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
        long yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address();

        long z = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
        long zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address();
        long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(op.getDimension()), context).address();

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

        allocator.registerAction(context, op.z(), op.x(), op.y());

        return null;
    }



    private CudaContext invoke(IndexAccumulation op,int[] dimension)  {

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        //log.info("OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");
        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;

        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()),
                context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar()};

      //  System.out.println("X shapeInfo host address: " + xShapeInfoHostPointer[0]);
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
            if (dimension == null)
                dimension = new int[] {0};

            Arrays.sort(dimension);

            long z = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
            long zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address();
            long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context).address();

//            log.info("Z.length: " + op.z().length());
//            log.info("Z.shapeInfo: " + op.z().shapeInfoDataBuffer());

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

        allocator.registerAction(context, op.z(), op.x(), op.y());

        return null;

    }


    private CudaContext invoke(Accumulation op, int[] dimension) {

    //    log.info("A OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");
        // dimension is ALWAYS null here.
        if (dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};

        Arrays.sort(dimension);

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar()};
        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;

        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[]{1, retShape[0]};
            else
                retShape = new int[]{retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[]{1, 1};
        }

        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return null;

        INDArray ret = null;
        if (op.zeroDouble() > -0.01f && op.zeroDouble() < 0.01f) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }
        op.setZ(ret);

        if(op.z().isScalar()) {
            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op instanceof Variance) {
                    double result = nativeOps.execSummaryStatsScalarDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x
                            ,xShapeInfo,extraArgs, true);
                    op.setFinalResult(result);
                } else if (op.y() != null) {
                    long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
                    long yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address();
                    double result = nativeOps.execReduce3ScalarDouble(
                            xShapeInfoHostPointer,
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
                    long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
                    long yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address();

                    float result = nativeOps.execReduce3ScalarFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
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

        }
        else {
            long result = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
            long resultShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address();
            long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context).address();

            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {
                    long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
                    long yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address();
                    nativeOps.execReduce3Double(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            y,
                            yShapeInfo,
                            result,
                            resultShapeInfo,
                            dimensionPointer,
                            dimension.length);
                }
                else {
                    if(op instanceof Variance) {
                       nativeOps.execSummaryStatsDouble(
                               xShapeInfoHostPointer,
                               op.opNum(),
                               x,
                               xShapeInfo,
                               extraArgs,
                               result,
                               resultShapeInfo,
                               dimensionPointer,
                               dimension.length,
                               ((Variance) op).isBiasCorrected());
                    }
                    else {
                        nativeOps.execReduceDouble(
                                xShapeInfoHostPointer,
                                op.opNum(),
                                x,
                                xShapeInfo,
                                extraArgs,
                                result,
                                resultShapeInfo,
                                dimensionPointer,
                                dimension.length);
                    }
                }

            }
            //float
            else {
                if(op.y() != null) {
                    long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
                    long yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address();
                    nativeOps.execReduce3Float(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            extraArgs,
                            y,
                            yShapeInfo,
                            result,
                            resultShapeInfo,
                            dimensionPointer,
                            dimension.length);

                }
                else {

                    if(op instanceof Variance) {
                        nativeOps.execSummaryStatsFloat(
                                xShapeInfoHostPointer,
                                op.opNum(),
                                x,
                                xShapeInfo,
                                extraArgs,
                                result,
                                resultShapeInfo,
                                dimensionPointer,
                                dimension.length,
                                ((Variance) op).isBiasCorrected());
                    }
                    else {
                        nativeOps.execReduceFloat(
                                xShapeInfoHostPointer,
                                op.opNum(),
                                x,
                                xShapeInfo,
                                extraArgs,
                                result,
                                resultShapeInfo,
                                dimensionPointer,
                                dimension.length);
                    }
                }

            }

        }

//&& !op.z().isScalar()
        allocator.registerAction(context, op.z(), op.x(), op.y());

        return context;
    }


    private CudaContext invoke(ScalarOp op) {
      //  log.info("OpName: [" + op.getClass().getSimpleName() + "]; OpCode: [" + op.opNum() + "]");

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;

        long z = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
        long zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address();
        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar() };

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

        allocator.registerAction(context, op.z(), op.x(), op.y());

        return  null;
    }

    private CudaContext invoke(TransformOp op) {
//        log.info("T OpName: [" + op.getClass().getCanonicalName() + "]; OpCode: [" + op.opNum() + "]");

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        long x = AtomicAllocator.getInstance().getPointer(op.x(), context).address();
        long xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context).address();
        long extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context).address() : 0;

        long z = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
        long zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context).address();
        long[] xShapeInfoHostPointer = new long[]{AddressRetriever.retrieveHostAddress(op.x().shapeInfoDataBuffer()), context.getOldStream().getNativePointer(), allocator.getDeviceId(), context.getBufferAllocation(), context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial()};
/*
        log.info("------------------------------------");
        log.info("xShapeInfoHostPointer: " + Arrays.toString(xShapeInfoHostPointer));
        log.info("X: {}, Y: {}, Z: {}", x, op.y() != null ? AtomicAllocator.getInstance().getPointer(op.y()).address() : null, z);
        log.info("xShapeInfo: " + xShapeInfo);
*/
        if(op.y() != null) {
            long y = AtomicAllocator.getInstance().getPointer(op.y(), context).address();
            long yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context).address();
/*
            log.info("X shapeInfo: " + op.x().shapeInfoDataBuffer());
            log.info("Y shapeInfo: " + op.y().shapeInfoDataBuffer());
            log.info("Z shapeInfo: " + op.z().shapeInfoDataBuffer());
*/

            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.x().elementWiseStride() >=1 && op.y().elementWiseStride() >= 1 && !op.isExecSpecial() && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) {
                    nativeOps.execPairwiseTransformDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            op.x().elementWiseStride(),
                            y,
                            op.y().elementWiseStride(),
                            z,
                            op.z().elementWiseStride(),
                            extraArgs,
                            op.n()
                    );
                } else {
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
                }
            } else {
                if(op.x().elementWiseStride() >=1 && op.y().elementWiseStride() >= 1 && !op.isExecSpecial() && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) {
                    nativeOps.execPairwiseTransformFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            op.x().elementWiseStride(),
                            y,
                            op.y().elementWiseStride(),
                            z,
                            op.z().elementWiseStride(),
                            extraArgs,
                            op.n()
                    );
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
        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.x().elementWiseStride() >=1 && op.y().elementWiseStride() >= 1 && !op.isExecSpecial() && op.x().ordering() == op.z().ordering()) {
                    nativeOps.execTransformDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            op.x().elementWiseStride(),
                            z,
                            op.z().elementWiseStride(),
                            extraArgs,
                            op.n()
                    );
                } else {
                    nativeOps.execTransformDouble(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            xShapeInfo,
                            z,
                            zShapeInfo,
                            extraArgs);
                }
            } else {
                if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.z().ordering() == op.x().ordering()) {
                    nativeOps.execTransformFloat(
                            xShapeInfoHostPointer,
                            op.opNum(),
                            x,
                            op.x().elementWiseStride(),
                            z,
                            op.z().elementWiseStride(),
                            extraArgs,
                            op.n()
                    );
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
        }


        allocator.registerAction(context, op.z(), op.x(), op.y());

        return null;
    }
}


