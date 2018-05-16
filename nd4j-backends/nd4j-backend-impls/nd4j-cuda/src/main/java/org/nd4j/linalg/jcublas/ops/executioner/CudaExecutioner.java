/*-
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


import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.tad.DeviceTADManager;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.CopyOp;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.LongPointerWrapper;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jCuda;

import java.util.*;


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
@Slf4j
public class CudaExecutioner extends DefaultOpExecutioner {

    protected static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    //    private static final Allocator allocator = AtomicAllocator.getInstance();

    @Getter
    protected static TADManager tadManager = new DeviceTADManager();
    protected ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();
    protected volatile transient Properties properties;

    protected ThreadLocal<String> lastOp = new ThreadLocal<>();

    protected Map<String, CustomOpDescriptor> customOps = null;

    public CudaExecutioner() {

    }

    public NativeOps getNativeOps() {
        return nativeOps;
    }

    @Override
    public String getLastOp() {
        return lastOp.get();
    }

    @Override
    public INDArray exec(BroadcastOp op, int... dimension) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        Arrays.sort(dimension);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
        Pointer z = AtomicAllocator.getInstance().getPointer(op.z(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);
        //        }

        // extraz.get().put
        // new PointerPointer
        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                devTadShapeInfoZ, devTadOffsetsZ);

        //Pointer dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);
        Pointer dimensionPointer = AtomicAllocator.getInstance()
                .getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context);

        if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execBroadcastDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x, (IntPointer) xShapeInfo,
                    (DoublePointer) y,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                            context),
                    (DoublePointer) z, (IntPointer) AtomicAllocator.getInstance()
                            .getPointer(op.z().shapeInfoDataBuffer(), context),
                    (IntPointer) dimensionPointer, dimension.length);
        } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.execBroadcastFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x, (IntPointer) xShapeInfo,
                    (FloatPointer) y,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                            context),
                    (FloatPointer) z, (IntPointer) AtomicAllocator.getInstance()
                            .getPointer(op.z().shapeInfoDataBuffer(), context),
                    (IntPointer) dimensionPointer, dimension.length);
        } else {
            nativeOps.execBroadcastHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x, (IntPointer) xShapeInfo,
                    (ShortPointer) y,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                            context),
                    (ShortPointer) z, (IntPointer) AtomicAllocator.getInstance()
                            .getPointer(op.z().shapeInfoDataBuffer(), context),
                    (IntPointer) dimensionPointer, dimension.length);
        }

        AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return op.z();
    }

    /**
     *
     * @param op
     * @param dimension
     * @return
     */
    protected INDArray naiveExec(Accumulation op, int... dimension) {
        long st = profilingHookIn(op);
        INDArray ret = op.z();

        validateDataType(Nd4j.dataType(), op);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);
/*
        if (op.opNum() == 3) {
            log.info("Max shape: {}", Arrays.toString(op.x().shapeInfoDataBuffer().asInt()));
            log.info("Max TAD: {}", Arrays.toString(tadBuffers.getFirst().asInt()));
            context.syncOldStream();
        }
*/
        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()),
                context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(),
                context.getBufferAllocation(),
                context.getBufferReduction(),
                context.getBufferScalar(),
                context.getBufferSpecial(),
                hostYShapeInfo,
                hostZShapeInfo,
                hostTadShapeInfo,
                devTadShapeInfo,
                devTadOffsets);

        Pointer yDevTadOffsets = null;
        Pointer yDevTadShapeInfo = null;

        if (op.y() != null) {
            if ((dimension.length == 1 &&  dimension[0] == Integer.MAX_VALUE )||op.x().tensorAlongDimension(0, dimension).lengthLong() != op.y().lengthLong()) {
                if (!op.isComplexAccumulation() && op.x().lengthLong() != op.y().lengthLong())
                    throw new ND4JIllegalStateException("Op.X [" + op.x().lengthLong() + "] and Op.Y [" + op.y().lengthLong() + "] lengths should match");


                Pair<DataBuffer, DataBuffer> yTadBuffers = tadManager.getTADOnlyShapeInfo(op.y(), dimension);

                yDevTadShapeInfo = AtomicAllocator.getInstance().getPointer(yTadBuffers.getFirst(), context);

                DataBuffer yOffsets = yTadBuffers.getSecond();
                yDevTadOffsets = yOffsets == null ? null : AtomicAllocator.getInstance().getPointer(yOffsets, context);

                xShapeInfoHostPointer.put(12, yDevTadShapeInfo);
                xShapeInfoHostPointer.put(13, yDevTadOffsets);
            } else {
                // TAD vs full array code branch
                val fakeOffsets = Nd4j.getConstantHandler().getConstantBuffer(new int[] {0, 0});
                yDevTadOffsets = fakeOffsets == null ? null : AtomicAllocator.getInstance().getPointer(fakeOffsets, context);

                yDevTadShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);

                xShapeInfoHostPointer.put(12, AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context));
                xShapeInfoHostPointer.put(13, null);
            }
        }


        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : null;
        //Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : 0;
        //Pointer dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);
        Pointer dimensionPointer = AtomicAllocator.getInstance()
                .getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context); //AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);

        if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            if (op instanceof Variance) {
                if (ret.isScalar()) {
                    double res = nativeOps.execSummaryStatsScalarDouble(xShapeInfoHostPointer, op.opNum(),
                            (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            ((Variance) op).isBiasCorrected());

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execSummaryStatsDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length, ((Variance) op).isBiasCorrected());

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            } else if (op.y() != null) {
                if (op.isComplexAccumulation()) {

                    val dT = new LongPointerWrapper(devTadOffsets);
                    val yT = new LongPointerWrapper(yDevTadOffsets);

                    nativeOps.execReduce3AllDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),context),
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context),
                            (IntPointer) dimensionPointer, dimension.length,
                            (IntPointer) devTadShapeInfo,
                            dT,
                            (IntPointer) yDevTadShapeInfo,
                            yT);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                } else if (ret.isScalar()) {
                    double res = nativeOps.execReduce3ScalarDouble(xShapeInfoHostPointer, op.opNum(),
                            (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                    context));
                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execReduce3Double(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                    context),
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            } else {
                if (ret.isScalar()) {
                    double res = nativeOps.execReduceScalarDouble(xShapeInfoHostPointer, op.opNum(),
                            (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execReduceDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            }
        } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            if (op instanceof Variance) {
                if (ret.isScalar()) {
                    float res = nativeOps.execSummaryStatsScalarFloat(xShapeInfoHostPointer, op.opNum(),
                            (FloatPointer) x, (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            ((Variance) op).isBiasCorrected());

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execSummaryStatsFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length, ((Variance) op).isBiasCorrected());

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            } else if (op.y() != null) {
                if (op.isComplexAccumulation()) {
                    nativeOps.execReduce3AllFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),context),
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context),
                            (IntPointer) dimensionPointer, dimension.length,
                            (IntPointer) devTadShapeInfo,
                            new LongPointerWrapper(devTadOffsets),
                            (IntPointer) yDevTadShapeInfo,
                            new LongPointerWrapper(yDevTadOffsets));

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                } else if (ret.isScalar()) {
                    float res = nativeOps.execReduce3ScalarFloat(xShapeInfoHostPointer, op.opNum(),
                            (FloatPointer) x, (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                    context));

                    ret.assign(res);
                    op.setFinalResult(res);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                } else {
                    nativeOps.execReduce3Float(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                    context),
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            } else {
                if (ret.isScalar()) {
                    float res = nativeOps.execReduceScalarFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execReduceFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            }
        } else {
            if (op instanceof Variance) {
                if (ret.isScalar()) {
                    float res = nativeOps.execSummaryStatsScalarHalf(xShapeInfoHostPointer, op.opNum(),
                            (ShortPointer) x, (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            ((Variance) op).isBiasCorrected());

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execSummaryStatsHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length, ((Variance) op).isBiasCorrected());

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            } else if (op.y() != null) {
                if (op.isComplexAccumulation()) {
                    nativeOps.execReduce3AllHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),context),
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context),
                            (IntPointer) dimensionPointer, dimension.length,
                            (IntPointer) devTadShapeInfo,
                            new LongPointerWrapper(devTadOffsets),
                            (IntPointer) yDevTadShapeInfo,
                            new LongPointerWrapper(yDevTadOffsets));

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                } else if (ret.isScalar()) {
                    float res = nativeOps.execReduce3ScalarHalf(xShapeInfoHostPointer, op.opNum(),
                            (ShortPointer) x, (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                    context));
                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                    ret.assign(res);

                    op.setFinalResult(res);
                } else {
                    nativeOps.execReduce3Half(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                    context),
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            } else {
                if (ret.isScalar()) {
                    float res = nativeOps.execReduceScalarHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                    ret.assign(res);
                    op.setFinalResult(res);
                } else {
                    nativeOps.execReduceHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                            (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                    context),
                            (IntPointer) dimensionPointer, dimension.length);

                    AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());
                }
            }
        }

        profilingHookOut(op, st);

        return op.z();
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        long st = profilingHookIn(op);
        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);


        Arrays.sort(dimension);


        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        int[] maxShape = Shape.getMaxShape(op.x(),op.y());
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= maxShape.length && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        for (int i = 0; i < dimension.length; i++) {
            if (dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        int[] retShape;
        if (Shape.wholeArrayDimension(dimension))
            retShape = new int[] {1, 1};
        else
            retShape = ArrayUtil.removeIndex(maxShape, dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[] {1, retShape[0]};
            else
                retShape = new int[] {retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[] {1, 1};
        }

        if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape) && ArrayUtil.prodLong(retShape) > 1 && op.y() == null)
            return op.noOp();

        INDArray ret = null;
        if (op.z() == null || op.z() == op.x()) {
            if (op.isComplexAccumulation()) {
                int xT = op.x().tensorssAlongDimension(dimension);
                int yT = op.y().tensorssAlongDimension(dimension);

                ret = Nd4j.create(xT, yT);
            } else {
                if (op.y() != null) {
                    val xT = op.x().tensorAlongDimension(0, dimension).lengthLong();
                    val yT = op.y().lengthLong();

                    if (xT != yT)
                        throw new ND4JIllegalStateException("Number of TADs along dimension doesn't match");
                }


                if (0.0 + Math.abs(op.zeroDouble()) <= Nd4j.EPS_THRESHOLD) {
                    ret = Nd4j.zeros(retShape);
                } else {
                    if (op.x().data().dataType() == DataBuffer.Type.DOUBLE)
                        ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
                    else if (op.x().data().dataType() == DataBuffer.Type.FLOAT)
                        ret = Nd4j.valueArrayOf(retShape, op.zeroFloat());
                    else if (op.x().data().dataType() == DataBuffer.Type.HALF)
                        ret = Nd4j.valueArrayOf(retShape, op.zeroHalf());
                }
            }
            op.setZ(ret);
        } else {
            // compare length
            if (op.z().lengthLong() != ArrayUtil.prodLong(retShape))
                throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + Arrays.toString(retShape) + "]");

            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                op.z().assign(op.zeroDouble());
            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                op.z().assign(op.zeroFloat());
            } else if (op.x().data().dataType() == DataBuffer.Type.HALF) {
                op.z().assign(op.zeroHalf());
            }

            ret = op.z();
        }

        naiveExec(op, dimension);


        profilingHookOut(op, st);

        return op.z();
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        Arrays.sort(dimension);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        for (int i = 0; i < dimension.length; i++) {
            if (dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};

        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1, 1}
                : ArrayUtil.removeIndex(op.x().shape(), dimension);


        if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape)) {
            return op.x();
        }


        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[] {1, retShape[0]};
            else
                retShape = new int[] {retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[] {1, 1};
        }

        INDArray ret = null;
        if (0.0 + Math.abs(op.zeroDouble()) <= Nd4j.EPS_THRESHOLD) {
            ret = Nd4j.zeros(retShape);
        } else {
            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE)
                ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
            else if (op.x().data().dataType() == DataBuffer.Type.FLOAT)
                ret = Nd4j.valueArrayOf(retShape, op.zeroFloat());
            else if (op.x().data().dataType() == DataBuffer.Type.HALF)
                ret = Nd4j.valueArrayOf(retShape, op.zeroHalf());
        }

        op.setZ(ret);
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);

        Pointer z = AtomicAllocator.getInstance().getPointer(op.z(), context);
        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);
        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : null;
        //Pointer dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);
        Pointer dimensionPointer = AtomicAllocator.getInstance()
                .getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context);

        if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execIndexReduceDouble(xShapeInfoHostPointer,
                    op.opNum(), (DoublePointer) x,
                    (IntPointer) xShapeInfo,
                    (DoublePointer) extraArgs,
                    (DoublePointer) z,
                    (IntPointer) zShapeInfo,
                    (IntPointer) dimensionPointer,
                    dimension.length);

        } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.execIndexReduceFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x, (IntPointer) xShapeInfo,
                    (FloatPointer) extraArgs, (FloatPointer) z, (IntPointer) zShapeInfo,
                    (IntPointer) dimensionPointer, dimension.length);

        } else {
            nativeOps.execIndexReduceHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x, (IntPointer) xShapeInfo,
                    (ShortPointer) extraArgs, (ShortPointer) z, (IntPointer) zShapeInfo,
                    (IntPointer) dimensionPointer, dimension.length);

        }

        AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return op.z();
    }


    @Override
    public Op exec(Op op, int... dimension) {
        checkForCompression(op);

        Arrays.sort(dimension);
        return super.exec(op, dimension);
    }


    @Override
    public Op exec(Op op) {
        checkForCompression(op);

        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpreted as vectors)
        if (op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA || op instanceof CopyOp) {
            // we dont' care about op.Z sync state, since it'll be overwritten
            if (op.x() != null)
                AtomicAllocator.getInstance().synchronizeHostData(op.x());
            if (op.y() != null)
                AtomicAllocator.getInstance().synchronizeHostData(op.y());

            super.exec(op);

            if (op.z() != null)
                AtomicAllocator.getInstance().tickHostWrite(op.z());
            return null;
        }

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc, null);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc);
        } else if (op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            invoke(broadcastOp);
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) op;
            invoke(indexAccumulation, null);
        }
        return op;
    }



    @Override
    public INDArray execAndReturn(TransformOp op) {
        checkForCompression(op);

        invoke(op);
        return op.z();
    }



    protected CudaContext invoke(BroadcastOp op) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), op.getDimension());

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), op.getDimension());

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                devTadShapeInfoZ, devTadOffsetsZ);

        Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
        Pointer yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);

        Pointer z = AtomicAllocator.getInstance().getPointer(op.z(), context);
        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);
        //long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(op.getDimension()), context);
        Pointer dimensionPointer = AtomicAllocator.getInstance()
                .getPointer(AtomicAllocator.getInstance().getConstantBuffer(op.getDimension()), context);

        if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execBroadcastDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x, (IntPointer) xShapeInfo,
                    (DoublePointer) y, (IntPointer) yShapeInfo, (DoublePointer) z, (IntPointer) zShapeInfo,
                    (IntPointer) dimensionPointer, op.getDimension().length);
        } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.execBroadcastFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x, (IntPointer) xShapeInfo,
                    (FloatPointer) y, (IntPointer) yShapeInfo, (FloatPointer) z, (IntPointer) zShapeInfo,
                    (IntPointer) dimensionPointer, op.getDimension().length);

        } else {
            nativeOps.execBroadcastHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x, (IntPointer) xShapeInfo,
                    (ShortPointer) y, (IntPointer) yShapeInfo, (ShortPointer) z, (IntPointer) zShapeInfo,
                    (IntPointer) dimensionPointer, op.getDimension().length);
        }

        AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return null;
    }



    protected CudaContext invoke(IndexAccumulation op, int[] dimension) {
        long st = profilingHookIn(op);

        if (dimension == null || (dimension.length == 1 && dimension[0] == Integer.MAX_VALUE)) {
            if(op.z() == op.x() || op.z() == null) {
                op.setZ(Nd4j.scalar(0.0));
            }
        }

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());
        CudaEnvironment.getInstance().getConfiguration().enableDebug(true);
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z().isScalar() ? null : op.z(), op.x(), op.y());

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);
        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : null;

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        int fdimension[] = dimension;
        if (fdimension == null)
            fdimension = new int[] {0};

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), fdimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);

        if (op.z().isScalar() || dimension == null || dimension[0] == Integer.MAX_VALUE) {
            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                double result = nativeOps.execIndexReduceScalarDouble(xShapeInfoHostPointer, op.opNum(),
                        (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs);
                op.setFinalResult((int) result);
                op.z().assign(result);

            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                float result = nativeOps.execIndexReduceScalarFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                        (IntPointer) xShapeInfo, (FloatPointer) extraArgs);
                op.setFinalResult((int) result);
                op.z().assign(result);
            } else {
                float result = nativeOps.execIndexReduceScalarHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                        (IntPointer) xShapeInfo, (ShortPointer) extraArgs);
                op.setFinalResult((int) result);
                op.z().assign(result);
            }

        } else {
            Arrays.sort(dimension);

            Pointer z = AtomicAllocator.getInstance().getPointer(op.z(), context);
            Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);
            //long dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);
            Pointer dimensionPointer = AtomicAllocator.getInstance()
                    .getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context);

            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execIndexReduceDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                        (IntPointer) xShapeInfo, (DoublePointer) extraArgs, (DoublePointer) z,
                        (IntPointer) zShapeInfo, (IntPointer) dimensionPointer, dimension.length);
            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                nativeOps.execIndexReduceFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                        (IntPointer) xShapeInfo, (FloatPointer) extraArgs, (FloatPointer) z,
                        (IntPointer) zShapeInfo, (IntPointer) dimensionPointer, dimension.length);
            } else {
                nativeOps.execIndexReduceHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                        (IntPointer) xShapeInfo, (ShortPointer) extraArgs, (ShortPointer) z,
                        (IntPointer) zShapeInfo, (IntPointer) dimensionPointer, dimension.length);
            }

        }


        AtomicAllocator.getInstance().registerAction(context, null, op.x(), op.y());

        profilingHookOut(op, st);

        return null;

    }


    protected CudaContext invoke(Accumulation op, int[] dimension) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        // dimension is ALWAYS null here.
        if (dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};

        Arrays.sort(dimension);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);

        if (op.y() != null) {
            Pair<DataBuffer, DataBuffer> yTadBuffers = tadManager.getTADOnlyShapeInfo(op.y(), dimension);

            Pointer yDevTadShapeInfo = AtomicAllocator.getInstance().getPointer(yTadBuffers.getFirst(), context);

            DataBuffer yOffsets = yTadBuffers.getSecond();
            Pointer yDevTadOffsets =
                    yOffsets == null ? null : AtomicAllocator.getInstance().getPointer(yOffsets, context);

            xShapeInfoHostPointer.put(12, yDevTadShapeInfo);
            xShapeInfoHostPointer.put(13, yDevTadOffsets);
        }

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);
        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : null;

        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1, 1}
                : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[] {1, retShape[0]};
            else
                retShape = new int[] {retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[] {1, 1};
        }

        if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return null;

        INDArray ret = null;
        if (0.0 + Math.abs(op.zeroDouble()) <= Nd4j.EPS_THRESHOLD) {
            ret = Nd4j.zeros(retShape);
        } else {
            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE)
                ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
            else if (op.x().data().dataType() == DataBuffer.Type.FLOAT)
                ret = Nd4j.valueArrayOf(retShape, op.zeroFloat());
            else if (op.x().data().dataType() == DataBuffer.Type.HALF)
                ret = Nd4j.valueArrayOf(retShape, op.zeroHalf());
        }
        op.setZ(ret);

        if (op.z().isScalar()) {
            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if (op instanceof Variance) {
                    double result = nativeOps.execSummaryStatsScalarDouble(xShapeInfoHostPointer, op.opNum(),
                            (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            ((Variance) op).isBiasCorrected());
                    op.setFinalResult(result);
                } else if (op.y() != null) {
                    Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
                    Pointer yShapeInfo =
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);
                    double result = nativeOps.execReduce3ScalarDouble(xShapeInfoHostPointer, op.opNum(),
                            (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs,
                            (DoublePointer) y, (IntPointer) yShapeInfo);
                    op.setFinalResult(result);
                } else {
                    double result = nativeOps.execReduceScalarDouble(xShapeInfoHostPointer, op.opNum(),
                            (DoublePointer) x, (IntPointer) xShapeInfo, (DoublePointer) extraArgs);
                    op.setFinalResult(result);
                }
            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                if (op instanceof Variance) {
                    float result = nativeOps.execSummaryStatsScalarFloat(xShapeInfoHostPointer, op.opNum(),
                            (FloatPointer) x, (IntPointer) xShapeInfo, (FloatPointer) extraArgs,
                            ((Variance) op).isBiasCorrected());
                    op.setFinalResult(result);
                } else if (op.y() != null) {
                    Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
                    Pointer yShapeInfo =
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);

                    float result = nativeOps.execReduce3ScalarFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs, (FloatPointer) y,
                            (IntPointer) yShapeInfo);
                    op.setFinalResult(result);
                } else {
                    float result = nativeOps.execReduceScalarFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs);
                    op.setFinalResult(result);
                }
            } else {
                if (op instanceof Variance) {
                    float result = nativeOps.execSummaryStatsScalarHalf(xShapeInfoHostPointer, op.opNum(),
                            (ShortPointer) x, (IntPointer) xShapeInfo, (ShortPointer) extraArgs,
                            ((Variance) op).isBiasCorrected());
                    op.setFinalResult(result);
                } else if (op.y() != null) {
                    Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
                    Pointer yShapeInfo =
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);

                    float result = nativeOps.execReduce3ScalarHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs, (ShortPointer) y,
                            (IntPointer) yShapeInfo);
                    op.setFinalResult(result);
                } else {
                    float result = nativeOps.execReduceScalarHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs);
                    op.setFinalResult(result);
                }
            }

        } else {
            Pointer result = AtomicAllocator.getInstance().getPointer(op.z(), context);
            Pointer resultShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);
            Pointer dimensionPointer = AtomicAllocator.getInstance()
                    .getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context); //AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);

            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if (op.y() != null) {
                    Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
                    Pointer yShapeInfo =
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);
                    nativeOps.execReduce3Double(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) extraArgs, (DoublePointer) y,
                            (IntPointer) yShapeInfo, (DoublePointer) result, (IntPointer) resultShapeInfo,
                            (IntPointer) dimensionPointer, dimension.length);
                } else {
                    if (op instanceof Variance) {
                        nativeOps.execSummaryStatsDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                                (IntPointer) xShapeInfo, (DoublePointer) extraArgs, (DoublePointer) result,
                                (IntPointer) resultShapeInfo, (IntPointer) dimensionPointer, dimension.length,
                                ((Variance) op).isBiasCorrected());
                    } else {
                        nativeOps.execReduceDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                                (IntPointer) xShapeInfo, (DoublePointer) extraArgs, (DoublePointer) result,
                                (IntPointer) resultShapeInfo, (IntPointer) dimensionPointer, dimension.length);
                    }
                }

            }
            //float
            else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                if (op.y() != null) {
                    Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
                    Pointer yShapeInfo =
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);
                    nativeOps.execReduce3Float(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) extraArgs, (FloatPointer) y,
                            (IntPointer) yShapeInfo, (FloatPointer) result, (IntPointer) resultShapeInfo,
                            (IntPointer) dimensionPointer, dimension.length);

                } else {

                    if (op instanceof Variance) {
                        nativeOps.execSummaryStatsFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                                (IntPointer) xShapeInfo, (FloatPointer) extraArgs, (FloatPointer) result,
                                (IntPointer) resultShapeInfo, (IntPointer) dimensionPointer, dimension.length,
                                ((Variance) op).isBiasCorrected());
                    } else {
                        nativeOps.execReduceFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                                (IntPointer) xShapeInfo, (FloatPointer) extraArgs, (FloatPointer) result,
                                (IntPointer) resultShapeInfo, (IntPointer) dimensionPointer, dimension.length);
                    }
                }
            } // Half
            else {
                if (op.y() != null) {
                    Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
                    Pointer yShapeInfo =
                            AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);
                    nativeOps.execReduce3Half(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) extraArgs, (ShortPointer) y,
                            (IntPointer) yShapeInfo, (ShortPointer) result, (IntPointer) resultShapeInfo,
                            (IntPointer) dimensionPointer, dimension.length);

                } else {

                    if (op instanceof Variance) {
                        nativeOps.execSummaryStatsHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                                (IntPointer) xShapeInfo, (ShortPointer) extraArgs, (ShortPointer) result,
                                (IntPointer) resultShapeInfo, (IntPointer) dimensionPointer, dimension.length,
                                ((Variance) op).isBiasCorrected());
                    } else {
                        nativeOps.execReduceHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                                (IntPointer) xShapeInfo, (ShortPointer) extraArgs, (ShortPointer) result,
                                (IntPointer) resultShapeInfo, (IntPointer) dimensionPointer, dimension.length);
                    }
                }
            }

        }

        AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return context;
    }


    protected CudaContext intercept(ScalarOp op, int[] dimension) {
        long st = profilingHookIn(op);

        Arrays.sort(dimension);

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer y = AtomicAllocator.getInstance().getPointer(op.y(), context);
        Pointer z = AtomicAllocator.getInstance().getPointer(op.z(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);
        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);


        PointerPointer extraPointers = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                devTadShapeInfoZ, devTadOffsetsZ);

        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : null;

        Pointer dimensionPointer = AtomicAllocator.getInstance()
                .getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context);


        if (op.x().data().dataType() == DataBuffer.Type.HALF) {
            nativeOps.execScalarHalf(extraPointers, op.opNum(), (ShortPointer) x, (IntPointer) xShapeInfo,
                    (ShortPointer) z, (IntPointer) zShapeInfo, (ShortPointer) y, (ShortPointer) extraArgs,
                    (IntPointer) dimensionPointer, dimension.length);
        } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.execScalarFloat(extraPointers, op.opNum(), (FloatPointer) x, (IntPointer) xShapeInfo,
                    (FloatPointer) z, (IntPointer) zShapeInfo, (FloatPointer) y, (FloatPointer) extraArgs,
                    (IntPointer) dimensionPointer, dimension.length);
        } else if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execScalarDouble(extraPointers, op.opNum(), (DoublePointer) x, (IntPointer) xShapeInfo,
                    (DoublePointer) z, (IntPointer) zShapeInfo, (DoublePointer) y, (DoublePointer) extraArgs,
                    (IntPointer) dimensionPointer, dimension.length);
        }

        AtomicAllocator.getInstance().getFlowController().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return null;
    }

    protected CudaContext invoke(ScalarOp op) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (op.x().length() != op.z().length())
            throw new ND4JIllegalStateException("op.X length should be equal to op.Y length: ["
                    + Arrays.toString(op.x().shapeInfoDataBuffer().asInt()) + "] != ["
                    + Arrays.toString(op.z().shapeInfoDataBuffer().asInt()) + "]");

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        if (op.getDimension() != null) {
            intercept(op, op.getDimension());
            return null;
        }

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pointer x = AtomicAllocator.getInstance().getPointer(op.x(), context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);
        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(), context) : null;

        Pointer z = AtomicAllocator.getInstance().getPointer(op.z(), context);
        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, null, null);

        if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            if (op.x().elementWiseStride() >= 1 && op.z().ordering() == op.x().ordering()) {
                nativeOps.execScalarDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                        op.x().elementWiseStride(), (DoublePointer) z, op.z().elementWiseStride(),
                        op.scalar().doubleValue(), (DoublePointer) extraArgs, op.n());
            } else {
                nativeOps.execScalarDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                        (IntPointer) xShapeInfo, (DoublePointer) z, (IntPointer) zShapeInfo,
                        op.scalar().doubleValue(), (DoublePointer) extraArgs);
            }
        } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            if (op.x().elementWiseStride() >= 1 && op.z().ordering() == op.x().ordering()) {
                nativeOps.execScalarFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                        op.x().elementWiseStride(), (FloatPointer) z, op.z().elementWiseStride(),
                        op.scalar().floatValue(), (FloatPointer) extraArgs, op.n());
            } else {
                nativeOps.execScalarFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x, (IntPointer) xShapeInfo,
                        (FloatPointer) z, (IntPointer) zShapeInfo, op.scalar().floatValue(),
                        (FloatPointer) extraArgs);
            }
        } else {
            if (op.x().elementWiseStride() >= 1 && op.z().ordering() == op.x().ordering()) {
                nativeOps.execScalarHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                        op.x().elementWiseStride(), (ShortPointer) z, op.z().elementWiseStride(),
                        op.scalar().floatValue(), (ShortPointer) extraArgs, op.n());
            } else {
                nativeOps.execScalarHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x, (IntPointer) xShapeInfo,
                        (ShortPointer) z, (IntPointer) zShapeInfo, op.scalar().floatValue(),
                        (ShortPointer) extraArgs);
            }
        }

        AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return null;
    }

    protected CudaContext invoke(TransformOp op) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        AtomicAllocator allocator = AtomicAllocator.getInstance();

        // this is special case for assign
        /*
        if (op.opNum() == 16 && op.y() != null && !op.y().isView() && !op.x().isView() && !op.z().isView()
                && op.z().ordering() == op.y().ordering() && op.y().ordering() == op.x().ordering()
                && Arrays.equals(op.y().shape(), op.z().shape()) && Arrays.equals(op.y().stride(), op.z().stride())
                ) {
            AllocationPoint point = allocator.getAllocationPoint(op.y());
            AllocationPoint pointDst = allocator.getAllocationPoint(op.z());
            synchronized (point) {
        //                log.info("X: {}; Y: {}, Z: {}", op.x().ordering(), op.y().ordering(), op.z().ordering());
                CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        
                allocator.memcpyDevice(op.z().data(), allocator.getPointer(op.y(), context), op.y().length() * op.y().data().getElementSize(), 0, context);
                context.syncOldStream();
        
                return null;
            }
        }
        */

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        // Pow operations might be special
        if (op.opNum() == 7) {
            if (op.y() != null && op.y().isScalar()) {
                Nd4j.getExecutioner().commit();
                op.setY(Nd4j.valueArrayOf(op.x().shape(), op.y().getDouble(0)));
                Nd4j.getExecutioner().commit();
            }
        }

        CudaContext context = allocator.getFlowController().prepareAction(op.z(), op.x(), op.y());

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        // special temp array for IsMax along dimension
        INDArray ret = null;

        Pointer x = allocator.getPointer(op.x(), context);
        Pointer xShapeInfo = allocator.getPointer(op.x().shapeInfoDataBuffer(), context);
        Pointer extraArgs = op.extraArgs() != null ? allocator.getPointer(op.extraArgsDataBuff(), context) : null;


        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pointer dimensionDevPointer = null;
        Pointer dimensionHostPointer = null;
        Pointer retPointer = null;
        int dimension[] = null;

        if (op.opNum() == 41 && op.extraArgs() != null) {
            // for IsMax along dimension we need special temporary buffer
            dimension = new int[(int) op.extraArgs()[0]];

            for (int i = 0; i < dimension.length; i++) {
                dimension[i] = (int) op.extraArgs()[i + 1];
            }


            for (int i = 0; i < dimension.length; i++) {
                if (dimension[i] < 0)
                    dimension[i] += op.x().rank();
            }
            //do op along all dimensions
            if (dimension.length == op.x().rank())
                dimension = new int[] {Integer.MAX_VALUE};

            int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1, 1}
                    : ArrayUtil.removeIndex(op.x().shape(), dimension);

            //ensure vector is proper shape
            if (retShape.length == 1) {
                if (dimension[0] == 0)
                    retShape = new int[] {1, retShape[0]};
                else
                    retShape = new int[] {retShape[0], 1};
            } else if (retShape.length == 0) {
                retShape = new int[] {1, 1};
            }

            ret = Nd4j.zeros(retShape);

            // FIXME: this maybe misleading use of this particular pointer
            hostYShapeInfo = allocator.getPointer(ret.shapeInfoDataBuffer(), context);

            //dimensionPointer = AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);
            DataBuffer dimensionBuffer = allocator.getConstantBuffer(dimension);
            dimensionDevPointer = allocator.getPointer(dimensionBuffer, context);
            dimensionHostPointer = allocator.getHostPointer(dimensionBuffer);

            retPointer = allocator.getPointer(ret, context);
        }

        Pointer hostTadShapeInfo = null;
        Pointer devTadShapeInfo = null;

        Pointer hostMaxTadShapeInfo = null;
        Pointer devMaxTadShapeInfo = null;

        Pair<DataBuffer, DataBuffer> tadBuffers;
        Pair<DataBuffer, DataBuffer> tadMaxBuffers;

        Pointer devTadOffsets = null;
        Pointer devMaxTadOffsets = null;

        if (op.opNum() >= 38 && op.opNum() <= 41) {

            if (op.opNum() != 41) {
                tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), new int[] {0});
                tadMaxBuffers = tadManager.getTADOnlyShapeInfo(op.x(), new int[] {1});

                hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
                devTadShapeInfo = allocator.getPointer(tadBuffers.getFirst(), context);

                hostMaxTadShapeInfo = AddressRetriever.retrieveHostPointer(tadMaxBuffers.getFirst());
                devMaxTadShapeInfo = allocator.getPointer(tadMaxBuffers.getFirst(), context);

                DataBuffer offsets = tadBuffers.getSecond();
                devTadOffsets = offsets == null ? null : allocator.getPointer(offsets, context);

                DataBuffer maxOffsets = tadMaxBuffers.getSecond();
                devMaxTadOffsets = maxOffsets == null ? null : allocator.getPointer(maxOffsets, context);
            } else {
                tadBuffers = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

                hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
                devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

                DataBuffer offsets = tadBuffers.getSecond();
                devTadOffsets = offsets == null ? null : allocator.getPointer(offsets, context);
            }
        }

        Pointer z = allocator.getPointer(op.z(), context);
        Pointer zShapeInfo = allocator.getPointer(op.z().shapeInfoDataBuffer(), context);


        PointerPointer xShapeInfoHostPointer =
                extraz.get().put(AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), // 0
                        context.getOldStream(), // 1
                        allocator.getDeviceIdPointer(), // 2
                        context.getBufferAllocation(), // 3
                        context.getBufferReduction(), // 4
                        context.getBufferScalar(), // 5
                        context.getBufferSpecial(), // 6
                        hostYShapeInfo, // 7
                        hostZShapeInfo, // 8
                        hostTadShapeInfo, // 9
                        devTadShapeInfo, // 10
                        devTadOffsets, // 11
                        hostMaxTadShapeInfo, // 12
                        devMaxTadShapeInfo, // 13
                        devMaxTadOffsets, // 14
                        dimensionDevPointer, // special pointer for IsMax  // 15
                        dimensionHostPointer, // special pointer for IsMax  // 16
                        retPointer, // special pointer for IsMax // 17
                        new CudaPointer(dimension == null ? 0 : dimension.length));




        if (op.y() != null) {
            Pointer y = allocator.getPointer(op.y(), context);
            Pointer yShapeInfo = allocator.getPointer(op.y().shapeInfoDataBuffer(), context);

            int xEWS = op.x().elementWiseStride();
            int yEWS = op.y().elementWiseStride();
            int zEWS = op.z().elementWiseStride();

            boolean xRow = op.x().isRowVector();
            boolean yRow = op.y().isRowVector();
            boolean zRow = op.z().isRowVector();


            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if ((xEWS >= 1 && yEWS >= 1 && zEWS >= 1 && !op.isExecSpecial()
                        && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) || (xEWS >= 1 && yEWS == xEWS && zEWS == xEWS && xRow && yRow && zRow)) {

                    nativeOps.execPairwiseTransformDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            xEWS, (DoublePointer) y, yEWS,
                            (DoublePointer) z, zEWS, (DoublePointer) extraArgs, op.n());
                } else {
                    nativeOps.execPairwiseTransformDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) y, (IntPointer) yShapeInfo,
                            (DoublePointer) z, (IntPointer) zShapeInfo, (DoublePointer) extraArgs);
                }
            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                if ((xEWS >= 1 && yEWS >= 1
                        && xEWS == yEWS && !op.isExecSpecial()
                        && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) || (xEWS >= 1 && yEWS == xEWS && zEWS == xEWS && xRow && yRow && zRow)) {
                    nativeOps.execPairwiseTransformFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            xEWS, (FloatPointer) y, yEWS,
                            (FloatPointer) z, zEWS, (FloatPointer) extraArgs, op.n());
                } else {
                    nativeOps.execPairwiseTransformFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) y, (IntPointer) yShapeInfo,
                            (FloatPointer) z, (IntPointer) zShapeInfo, (FloatPointer) extraArgs);
                }
            } else {
                if ((xEWS >= 1 && yEWS >= 1
                        && xEWS == op.y().elementWiseStride() && !op.isExecSpecial()
                        && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) || (xEWS >= 1 && yEWS == xEWS && zEWS == xEWS && xRow && yRow && zRow)) {
                    nativeOps.execPairwiseTransformHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            xEWS, (ShortPointer) y, yEWS,
                            (ShortPointer) z, zEWS, (ShortPointer) extraArgs, op.n());
                } else {
                    nativeOps.execPairwiseTransformHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) y, (IntPointer) yShapeInfo,
                            (ShortPointer) z, (IntPointer) zShapeInfo, (ShortPointer) extraArgs);
                }
            }
        } else {
            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if (op.x().elementWiseStride() >= 1 && !op.isExecSpecial() && op.z().ordering() == op.x().ordering()) {
                    nativeOps.execTransformDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            op.x().elementWiseStride(), (DoublePointer) z, op.z().elementWiseStride(),
                            (DoublePointer) extraArgs, op.n());
                } else {
                    nativeOps.execTransformDouble(xShapeInfoHostPointer, op.opNum(), (DoublePointer) x,
                            (IntPointer) xShapeInfo, (DoublePointer) z, (IntPointer) zShapeInfo,
                            (DoublePointer) extraArgs);
                }
            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                if (op.x().elementWiseStride() >= 1 && !op.isExecSpecial() && op.z().ordering() == op.x().ordering()) {
                    nativeOps.execTransformFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            op.x().elementWiseStride(), (FloatPointer) z, op.z().elementWiseStride(),
                            (FloatPointer) extraArgs, op.n());
                } else {
                    nativeOps.execTransformFloat(xShapeInfoHostPointer, op.opNum(), (FloatPointer) x,
                            (IntPointer) xShapeInfo, (FloatPointer) z, (IntPointer) zShapeInfo,
                            (FloatPointer) extraArgs);
                }
            } else {
                if (op.x().elementWiseStride() >= 1 && !op.isExecSpecial() && op.z().ordering() == op.x().ordering()) {
                    nativeOps.execTransformHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            op.x().elementWiseStride(), (ShortPointer) z, op.z().elementWiseStride(),
                            (ShortPointer) extraArgs, op.n());
                } else {
                    nativeOps.execTransformHalf(xShapeInfoHostPointer, op.opNum(), (ShortPointer) x,
                            (IntPointer) xShapeInfo, (ShortPointer) z, (IntPointer) zShapeInfo,
                            (ShortPointer) extraArgs);
                }
            }
        }


        AtomicAllocator.getInstance().registerAction(context, op.z(), op.x(), op.y());

        if (extraArgs != null)
            extraArgs.address();

        if (ret != null)
            ret.elementWiseStride();

        profilingHookOut(op, st);

        return null;
    }

    protected <T extends Aggregate> DataBuffer getBuffer(Batch<T> batch) {
        DataBuffer buffer = Nd4j.getDataBufferFactory().createInt(batch.getSample().getRequiredBatchMemorySize() * 4,
                false);
        batch.setParamsSurface(buffer);
        return buffer;
    }

    @Override
    public <T extends Aggregate> void exec(Batch<T> batch) {
        DataBuffer surfaceBuffer = getBuffer(batch);

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        IntPointer pointer = (IntPointer) new CudaPointer(AtomicAllocator.getInstance().getHostPointer(surfaceBuffer))
                .asIntPointer();
        AllocationPoint surfacePoint = AtomicAllocator.getInstance().getAllocationPoint(surfaceBuffer);

        int maxTypes = 5;

        int maxIntArrays = batch.getSample().maxIntArrays();

        int maxArraySize = batch.getSample().maxIntArraySize();


        int indexPos = maxTypes * (Batch.getBatchLimit() * 16);
        int intArraysPos = indexPos + (batch.getSample().maxIndexArguments() * (Batch.getBatchLimit() * 16));
        int realPos = (intArraysPos + (maxIntArrays * maxArraySize * (Batch.getBatchLimit() * 16)))
                / (Nd4j.dataType() == DataBuffer.Type.DOUBLE ? 2 : 1);

        if (Nd4j.dataType() == DataBuffer.Type.HALF)
            realPos *= 2;

        int argsPos = (realPos + (batch.getSample().maxRealArguments() * (Batch.getBatchLimit() * 16)))
                / (Nd4j.dataType() == DataBuffer.Type.FLOAT ? 2 : 1);

        if (Nd4j.dataType() == DataBuffer.Type.HALF)
            argsPos /= 4;

        int shapesPos = argsPos + (batch.getSample().maxArguments() * (Batch.getBatchLimit() * 16));
        for (int i = 0; i < batch.getNumAggregates(); i++) {
            T op = batch.getAggregates().get(i);

            // put num arguments
            int idx = i * maxTypes;
            pointer.put(idx, op.getArguments().size());
            pointer.put(idx + 1, op.getShapes().size());
            pointer.put(idx + 2, op.getIndexingArguments().size());
            pointer.put(idx + 3, op.getRealArguments().size());
            pointer.put(idx + 4, op.getIntArrayArguments().size());


            // putting indexing arguments
            for (int e = 0; e < op.getIndexingArguments().size(); e++) {
                idx = indexPos + i * batch.getSample().maxIndexArguments();
                pointer.put(idx + e, op.getIndexingArguments().get(e));
            }

            // putting intArray values
            int bsize = maxIntArrays * maxArraySize;
            for (int e = 0; e < op.getIntArrayArguments().size(); e++) {
                int step = (i * bsize) + (e * maxArraySize);
                if (op.getIntArrayArguments().get(e) != null)
                    for (int x = 0; x < op.getIntArrayArguments().get(e).length; x++) {
                        idx = intArraysPos + step + x;
                        pointer.put(idx, op.getIntArrayArguments().get(e)[x]);
                    }
            }

            // TODO: variable datatype should be handled here
            // putting real arguments
            if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
                FloatPointer realPtr = new FloatPointer(pointer);
                for (int e = 0; e < op.getRealArguments().size(); e++) {
                    idx = realPos + i * op.maxRealArguments();
                    realPtr.put(idx + e, op.getRealArguments().get(e).floatValue());
                }
            } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
                DoublePointer dPtr = new DoublePointer(pointer);
                for (int e = 0; e < op.getRealArguments().size(); e++) {
                    idx = realPos + (i * op.maxRealArguments());
                    dPtr.put(idx + e, op.getRealArguments().get(e).doubleValue());
                }
            } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
                ShortPointer sPtr = new ShortPointer(pointer);
                for (int e = 0; e < op.getRealArguments().size(); e++) {
                    idx = realPos + (i * op.maxRealArguments());
                    sPtr.put(idx + e, BaseDataBuffer.fromFloat(op.getRealArguments().get(e).floatValue()));
                }
            }

            // putting arguments pointers
            PointerPointer ptrPtr = new PointerPointer(pointer);
            for (int e = 0; e < op.getArguments().size(); e++) {
                idx = argsPos + i * batch.getSample().maxArguments();

                if (op.getArguments().get(e) != null) {
                    ptrPtr.put(idx + e, AtomicAllocator.getInstance().getPointer(op.getArguments().get(e), context));
                    AtomicAllocator.getInstance().getAllocationPoint(op.getArguments().get(e)).tickDeviceWrite();
                }
            }


            // putting shape pointers
            for (int e = 0; e < op.getShapes().size(); e++) {
                idx = shapesPos + i * batch.getSample().maxShapes();

                if (op.getShapes().get(e) != null) {
                    ptrPtr.put(idx + e, AtomicAllocator.getInstance().getPointer(op.getShapes().get(e), context));
                    AtomicAllocator.getInstance().getAllocationPoint(op.getShapes().get(e)).tickDeviceWrite();
                }
            }
        }

        // trigger write, so getPointer request will force relocation to GPU
        surfacePoint.tickHostWrite();

        PointerPointer extraArgs = new PointerPointer(32);
        extraArgs.put(0, null);
        extraArgs.put(1, context.getOldStream());
        extraArgs.put(2, new CudaPointer(Math.min(batch.getNumAggregates(),
                CudaEnvironment.getInstance().getConfiguration().getMaximumGridSize())));
        extraArgs.put(3, new CudaPointer(batch.getSample().getThreadsPerInstance()));
        extraArgs.put(4, new CudaPointer(batch.getSample().getSharedMemorySize()));

        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.execAggregateBatchFloat(extraArgs, batch.getNumAggregates(), batch.opNum(),
                    batch.getSample().maxArguments(), batch.getSample().maxShapes(),
                    batch.getSample().maxIntArrays(), batch.getSample().maxIntArraySize(),
                    batch.getSample().maxIndexArguments(), batch.getSample().maxRealArguments(),
                    AtomicAllocator.getInstance().getPointer(surfaceBuffer, context));
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execAggregateBatchDouble(extraArgs, batch.getNumAggregates(), batch.opNum(),
                    batch.getSample().maxArguments(), batch.getSample().maxShapes(),
                    batch.getSample().maxIntArrays(), batch.getSample().maxIntArraySize(),
                    batch.getSample().maxIndexArguments(), batch.getSample().maxRealArguments(),
                    AtomicAllocator.getInstance().getPointer(surfaceBuffer, context));
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            nativeOps.execAggregateBatchHalf(extraArgs, batch.getNumAggregates(), batch.opNum(),
                    batch.getSample().maxArguments(), batch.getSample().maxShapes(),
                    batch.getSample().maxIntArrays(), batch.getSample().maxIntArraySize(),
                    batch.getSample().maxIndexArguments(), batch.getSample().maxRealArguments(),
                    AtomicAllocator.getInstance().getPointer(surfaceBuffer, context));
        }

        surfacePoint.tickHostWrite();
    }

    @Override
    public void exec(List<Aggregate> batch) {
        if (batch.size() == 0)
            return;

        List<Batch<Aggregate>> batches = Batch.getBatches(batch, 8192);
        for (Batch<Aggregate> single : batches) {
            this.exec(single);
        }

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        context.syncOldStream();
    }

    @Override
    public void exec(Aggregate op) {
        int numArguments = op.getArguments().size();
        int numShapeArguments = op.getShapes().size();
        int numIndexArguments = op.getIndexingArguments().size();
        int numIntArrays = op.getIntArrayArguments().size();
        int numRealArguments = op.getRealArguments().size();

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        PointerPointer extraArgs = new PointerPointer(32);
        extraArgs.put(0, null);
        extraArgs.put(1, context.getOldStream());
        extraArgs.put(2, new CudaPointer(1));
        extraArgs.put(3, new CudaPointer(op.getThreadsPerInstance()));
        extraArgs.put(4, new CudaPointer(op.getSharedMemorySize()));

        long arguments[] = new long[numArguments];

        for (int x = 0; x < numArguments; x++) {
            arguments[x] = op.getArguments().get(x) == null ? 0
                    : AtomicAllocator.getInstance().getPointer(op.getArguments().get(x), context).address();

            if (op.getArguments().get(x) != null)
                AtomicAllocator.getInstance().getAllocationPoint(op.getArguments().get(x)).tickDeviceWrite();
        }

        DataBuffer tempX = AllocationUtils.getPointersBuffer(arguments);
        PointerPointer xPtr = new PointerPointer(AtomicAllocator.getInstance().getPointer(tempX, context));


        long shapes[] = new long[numShapeArguments];
        for (int x = 0; x < numShapeArguments; x++) {
            shapes[x] = op.getShapes().get(x) == null ? 0
                    : AtomicAllocator.getInstance().getPointer(op.getShapes().get(x), context).address();

            if (op.getShapes().get(x) != null)
                AtomicAllocator.getInstance().getAllocationPoint(op.getShapes().get(x)).tickDeviceWrite();
        }

        DataBuffer tempS = AllocationUtils.getPointersBuffer(shapes);
        PointerPointer sPtr = new PointerPointer(AtomicAllocator.getInstance().getPointer(tempS, context));


        long ints[] = new long[numIntArrays];
        for (int x = 0; x < numIntArrays; x++) {
            if (op.getIntArrayArguments().get(x) != null) {
                DataBuffer intBuf = Nd4j.getDataBufferFactory().createInt(op.getIntArrayArguments().get(x));
                ints[x] = AtomicAllocator.getInstance().getPointer(intBuf, context).address();
            }

        }

        DataBuffer tempI = AllocationUtils.getPointersBuffer(ints);
        PointerPointer iPtr = new PointerPointer(AtomicAllocator.getInstance().getPointer(tempI, context));

        int[] indexes = new int[numIndexArguments];
        for (int x = 0; x < numIndexArguments; x++) {
            indexes[x] = op.getIndexingArguments().get(x);
        }

        DataBuffer intBuffer = Nd4j.getDataBufferFactory().createInt(indexes);

        double[] reals = new double[numRealArguments];
        for (int x = 0; x < numRealArguments; x++) {
            reals[x] = op.getRealArguments().get(x).doubleValue();
        }

        INDArray realsBuffer = Nd4j.create(reals);


        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.execAggregateFloat(extraArgs, op.opNum(), xPtr, numArguments, sPtr, numShapeArguments,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(intBuffer, context),
                    numIndexArguments, iPtr, numIntArrays,
                    (FloatPointer) AtomicAllocator.getInstance().getPointer(realsBuffer.data(), context),
                    numRealArguments);
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.execAggregateDouble(extraArgs, op.opNum(), xPtr, numArguments, sPtr, numShapeArguments,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(intBuffer, context),
                    numIndexArguments, iPtr, numIntArrays,
                    (DoublePointer) AtomicAllocator.getInstance().getPointer(realsBuffer.data(), context),
                    numRealArguments);
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            nativeOps.execAggregateHalf(extraArgs, op.opNum(), xPtr, numArguments, sPtr, numShapeArguments,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(intBuffer, context),
                    numIndexArguments, iPtr, numIntArrays,
                    (ShortPointer) AtomicAllocator.getInstance().getPointer(realsBuffer.data(), context),
                    numRealArguments);
        }
    }

    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return exec(op, Nd4j.getRandom());
    }


    @Override
    public INDArray exec(RandomOp op, Random rng) {
        long st = profilingHookIn(op);

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (rng.getStateBuffer() == null)
            throw new IllegalStateException(
                    "You should use one of NativeRandom classes for NativeOperations execution");

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());

        PointerPointer extraZZ = extraz.get().put(AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer()),
                context.getOldStream(), AtomicAllocator.getInstance().getDeviceIdPointer());

        if (op.x() != null && op.y() != null && op.z() != null) {
            // triple arg call
            if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
                nativeOps.execRandomFloat(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.x(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(),
                                context),
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                context),
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execRandomDouble(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.x(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(),
                                context),
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                context),
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
                nativeOps.execRandomHalf(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.x(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(),
                                context),
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.y(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(),
                                context),
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            }
        } else if (op.x() != null && op.z() != null) {
            //double arg call
            if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
                nativeOps.execRandomFloat(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.x(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(),
                                context),
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execRandomDouble(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.x(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(),
                                context),
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
                nativeOps.execRandomHalf(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.x(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(),
                                context),
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            }

        } else {
            // single arg call

            if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
                nativeOps.execRandomFloat(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (FloatPointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
                nativeOps.execRandomDouble(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (DoublePointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
                nativeOps.execRandomHalf(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.z(), context),
                        (IntPointer) AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(),
                                context),
                        (ShortPointer) AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(),
                                context));
            }
        }

        AtomicAllocator.getInstance().getFlowController().registerAction(context, op.z(), op.x(), op.y());

        profilingHookOut(op, st);

        return op.z();
    }

    /**
     * This method return set of key/value
     * and key/key/value objects,
     * describing current environment
     *
     * @return
     */
    @Override
    public synchronized Properties getEnvironmentInformation() {
        if (properties == null) {
            Properties props = super.getEnvironmentInformation();

            List<Map<String, Object>> devicesList = new ArrayList<>();

            // fill with per-device information: name, memory, versions
            for (int i = 0; i < nativeOps.getAvailableDevices(); i++) {
                Map<String, Object> deviceProps = new HashMap<>();

                CudaPointer devPtr = new CudaPointer(i);

                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY, nativeOps.getDeviceName(devPtr));
                deviceProps.put(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY, nativeOps.getDeviceFreeMemory(devPtr));
                deviceProps.put(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY, nativeOps.getDeviceTotalMemory(devPtr));
                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_MAJOR_VERSION_KEY, (long) nativeOps.getDeviceMajor(devPtr));
                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_MINOR_VERSION_KEY, (long) nativeOps.getDeviceMinor(devPtr));

                devicesList.add(i, deviceProps);
            }

            // fill with basic general info
            props.put(Nd4jEnvironment.BACKEND_KEY, "CUDA");
            props.put(Nd4jEnvironment.CUDA_NUM_GPUS_KEY, nativeOps.getAvailableDevices());
            props.put(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY, devicesList);
            props.put(Nd4jEnvironment.BLAS_VENDOR_KEY, (Nd4j.factory().blas()).getBlasVendor().toString());
            props.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

            // fill bandwidth information
            props.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());

            properties = props;
        } else {

            List<Map<String, Object>> devicesList = (List<Map<String, Object>>) properties.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);

            // just update information that might change over time
            for (int i = 0; i < nativeOps.getAvailableDevices(); i++) {
                Map<String, Object> dev = devicesList.get(i);
                CudaPointer devPtr = new CudaPointer(i);

                dev.put(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY, nativeOps.getDeviceFreeMemory(devPtr));
                dev.put(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY, nativeOps.getDeviceTotalMemory(devPtr));
            }

            properties.put(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY, devicesList);
            properties.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

            // fill bandwidth information
            properties.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());
        }
        return properties;
    }

    @Override
    public TADManager getTADManager() {
        return tadManager;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void printEnvironmentInformation() {
        super.printEnvironmentInformation();

        Properties env = getEnvironmentInformation();

        List<Map<String, Object>> devicesList = (List<Map<String, Object>>) env.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);
        for (Map<String, Object> dev : devicesList) {
            log.info("Device Name: [{}]; CC: [{}.{}]; Total/free memory: [{}]", dev.get(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY),
                    dev.get(Nd4jEnvironment.CUDA_DEVICE_MAJOR_VERSION_KEY), dev.get(Nd4jEnvironment.CUDA_DEVICE_MINOR_VERSION_KEY), dev.get(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY));
        }
    }

    @Override
    public void commit() {
        ((CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext()).syncOldStream();
        ((CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext()).syncSpecialStream();
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {
        DataBuffer buffer = input.data();

        int numThreads = 1024;
        int numBlocks = (int) (buffer.length() / numThreads + (buffer.length() % numThreads == 0 ? 0 : 1));

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        DataBuffer blocksBuffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createInt(numBlocks+1, true) : Nd4j.getDataBufferFactory().createInt(numBlocks+1, true, Nd4j.getMemoryManager().getCurrentWorkspace());

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer extras = extraz.get().put(1, context.getOldStream());


        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP1Float(extras, (FloatPointer) AtomicAllocator.getInstance().getPointer(buffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(blocksBuffer), (float) threshold);
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP1Double(extras, (DoublePointer) AtomicAllocator.getInstance().getPointer(buffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(blocksBuffer), (float) threshold);
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP1Half(extras, (ShortPointer) AtomicAllocator.getInstance().getPointer(buffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(blocksBuffer), (float) threshold);
        }


        AtomicAllocator.getInstance().getAllocationPoint(blocksBuffer).tickDeviceWrite();


        int numMatches = blocksBuffer.getInt(0);

        // special case here, nothing to update
        if (numMatches < 2)
            return null;

        if (boundary != null && numMatches > boundary)  {
            numMatches = boundary;
            blocksBuffer.put(0, numMatches);
        }

/*
        log.info("Totals: {}", numMatches);


        log.info("Number of blocks for compression: {}", numBlocks);
        log.info("BlocksCounts: {}", Arrays.toString(blocksBuffer.asInt()));
*/
        DataBuffer encodedBuffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createInt(4+numMatches, false) : Nd4j.getDataBufferFactory().createInt(4+numMatches, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        AtomicAllocator.getInstance().getAllocationPoint(encodedBuffer).tickHostWrite();
        encodedBuffer.put(0, numMatches);
        encodedBuffer.put(1, (int) buffer.length());
        encodedBuffer.put(2, Float.floatToIntBits((float) threshold));
        AtomicAllocator.getInstance().getAllocationPoint(encodedBuffer).tickHostWrite();

        encodedBuffer.put(3, ThresholdCompression.FLEXIBLE_ENCODING);


        int prefixThreads = 512;
        int numElts = numBlocks;
        int level = 0;
        List<DataBuffer> buffers = new ArrayList<>();

        // here we just calculate number of sumBlock arrays
        do {
            int numPrefixBlocks = Math.max(1, (int)Math.ceil((float)numElts / (2.0f * prefixThreads)));
            if (numBlocks > 1) {
                level++;
            }
            numElts = numPrefixBlocks;
        } while (numElts > 1);

        long[] pointers = new long[level];

        level = 0;
        numElts = numBlocks;

        //  allocating temp buffers for prefux sum
        DataBuffer tempX = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createDouble(pointers.length, false) : Nd4j.getDataBufferFactory().createDouble(pointers.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());

        do {
            int numPrefixBlocks = Math.max(1, (int)Math.ceil((float)numElts / (2.0f * prefixThreads)));
            if (numPrefixBlocks > 1) {
                DataBuffer bf = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createInt(numPrefixBlocks, false) : Nd4j.getDataBufferFactory().createInt(numPrefixBlocks, false, Nd4j.getMemoryManager().getCurrentWorkspace());

                buffers.add(bf);

                pointers[level++] = AtomicAllocator.getInstance().getPointer(bf).address();
            }
            numElts = numPrefixBlocks;
        } while (numElts > 1);


        AtomicAllocator.getInstance().memcpyBlocking(tempX, new LongPointer(pointers), pointers.length * 8, 0);

        extras.put(2, AtomicAllocator.getInstance().getPointer(tempX));

        DataBuffer offsetsBuffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createInt(numBlocks, true) : Nd4j.getDataBufferFactory().createInt(numBlocks, true, Nd4j.getMemoryManager().getCurrentWorkspace());

        NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP2Int(extras, (IntPointer) AtomicAllocator.getInstance().getPointer(blocksBuffer), numBlocks, (IntPointer) AtomicAllocator.getInstance().getPointer(offsetsBuffer) );
        AtomicAllocator.getInstance().getAllocationPoint(offsetsBuffer).tickDeviceWrite();

        //log.info("Offsets: {}", Arrays.toString(offsetsBuffer.asInt()));

        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP3Float(extras, (FloatPointer) AtomicAllocator.getInstance().getPointer(buffer), (IntPointer) AtomicAllocator.getInstance().getPointer(offsetsBuffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(encodedBuffer));
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP3Double(extras, (DoublePointer) AtomicAllocator.getInstance().getPointer(buffer), (IntPointer) AtomicAllocator.getInstance().getPointer(offsetsBuffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(encodedBuffer));
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP3Half(extras, (ShortPointer) AtomicAllocator.getInstance().getPointer(buffer), (IntPointer) AtomicAllocator.getInstance().getPointer(offsetsBuffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(encodedBuffer));
        }

        AtomicAllocator.getInstance().getAllocationPoint(encodedBuffer).tickDeviceWrite();
        AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();


        // just to ensure it's not purged
        extras.address();
        tempX.address();
        buffers.getClass();


        return Nd4j.createArrayFromShapeBuffer(encodedBuffer, input.shapeInfoDataBuffer());
    }


    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return thresholdEncode(input, threshold, null);
    }

    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        DataBuffer buffer = encoded.data();

        if (buffer.dataType() != DataBuffer.Type.INT)
            throw new UnsupportedOperationException();

        long compressedLength = buffer.getInt(0);
        long originalLength = buffer.getInt(1);

        if (target.lengthLong() != originalLength)
            throw new ND4JIllegalStateException("originalLength ["+ originalLength+"] stored in encoded array doesn't match target length ["+ target.lengthLong()+"]");

        DataBuffer result = target.data();



        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        //nativeOps.memsetAsync(AtomicAllocator.getInstance().getPointer(result), 0,result.length(), 0, context.getOldStream());

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer extras = extraz.get().put(1, context.getOldStream());

        //log.info("DEC Source length: {}", buffer.length());
        //log.info("DEC Source: {}", Arrays.toString(buffer.asInt()));

        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.decodeThresholdFloat(extras, AtomicAllocator.getInstance().getPointer(buffer), compressedLength, (FloatPointer) AtomicAllocator.getInstance().getPointer(result));
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.decodeThresholdDouble(extras, AtomicAllocator.getInstance().getPointer(buffer), compressedLength, (DoublePointer) AtomicAllocator.getInstance().getPointer(result));
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            nativeOps.decodeThresholdHalf(extras, AtomicAllocator.getInstance().getPointer(buffer), compressedLength, (ShortPointer) AtomicAllocator.getInstance().getPointer(result));
        }

        AtomicAllocator.getInstance().getAllocationPoint(result).tickDeviceWrite();


        //DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.THRESHOLD, buffer, getGlobalTypeEx());

        return target;
    }


    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        long length = indArray.lengthLong();
        long tLen = target.data().length();

        if (tLen != (length / 16 + 5))
            throw new ND4JIllegalStateException("Length of target array should be " + (length / 16 + 5));

        if (target.data().dataType() != DataBuffer.Type.INT)
            throw new ND4JIllegalStateException("Target array should have INT dataType");

        DataBuffer buffer = target.data();
        buffer.put(0, (int) length);
        buffer.put(1, (int) length);
        buffer.put(2, Float.floatToIntBits((float) threshold));

        // format id
        buffer.put(3, ThresholdCompression.BITMAP_ENCODING);

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(indArray);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));


        PointerPointer extras = extraz.get().put(
                AtomicAllocator.getInstance().getHostPointer(indArray),
                context.getOldStream(),
                context.getBufferScalar(),
                context.getBufferReduction()
        );

        long val = 0;
        if (indArray.data().dataType() == DataBuffer.Type.FLOAT) {
            val = nativeOps.encodeBitmapFloat(extras,
                    (FloatPointer) AtomicAllocator.getInstance().getPointer(indArray, context),
                    length,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(buffer, context),
                    (float) threshold
            );
        } else if (indArray.data().dataType() == DataBuffer.Type.DOUBLE) {
            val = nativeOps.encodeBitmapDouble(extras,
                    (DoublePointer) AtomicAllocator.getInstance().getPointer(indArray, context),
                    length,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(buffer, context),
                    (float) threshold
            );
        } else if (indArray.data().dataType() == DataBuffer.Type.HALF) {
            val = nativeOps.encodeBitmapHalf(extras,
                    (ShortPointer) AtomicAllocator.getInstance().getPointer(indArray, context),
                    length,
                    (IntPointer) AtomicAllocator.getInstance().getPointer(buffer, context),
                    (float) threshold
            );
        } else
            throw new ND4JIllegalStateException("Unknown dataType " + indArray.data().dataType());


        AtomicAllocator.getInstance().getFlowController().registerAction(context, indArray);

        AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();

        return val;
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(target);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));


        PointerPointer extras = extraz.get().put(
                AtomicAllocator.getInstance().getHostPointer(target),
                context.getOldStream(),
                context.getBufferScalar(),
                context.getBufferReduction());

        if (target.data().dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.decodeBitmapFloat(extras, AtomicAllocator.getInstance().getPointer(encoded.data(), context), target.lengthLong(), (FloatPointer) AtomicAllocator.getInstance().getPointer(target, context));
        } else if (target.data().dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.decodeBitmapDouble(extras, AtomicAllocator.getInstance().getPointer(encoded.data(), context), target.lengthLong(), (DoublePointer) AtomicAllocator.getInstance().getPointer(target, context));
        } else if (target.data().dataType() == DataBuffer.Type.HALF) {
            nativeOps.decodeBitmapHalf(extras, AtomicAllocator.getInstance().getPointer(encoded.data(), context), target.lengthLong(), (ShortPointer) AtomicAllocator.getInstance().getPointer(target, context));
        } else
            throw new ND4JIllegalStateException("Unknown dataType " + target.data().dataType());


        AtomicAllocator.getInstance().getFlowController().registerAction(context, target);

        return target;
    }


    @Override
    public synchronized Map<String, CustomOpDescriptor> getCustomOperations() {
        if(customOps == null) {
            String list = nativeOps.getAllCustomOps();

            if (list == null || list.isEmpty()) {
                log.warn("No customs ops available!");
                customOps = Collections.emptyMap();
                return customOps;
            }

            val map = new HashMap<String, CustomOpDescriptor>();

            String[] split = list.split(";");
            for (String op : split) {
                if (op == null || op.isEmpty())
                    continue;

                String[] another = op.split(":");

                CustomOpDescriptor descriptor = CustomOpDescriptor.builder()
                        .hash(Long.valueOf(another[1]))
                        .numInputs(Integer.valueOf(another[2]))
                        .numOutputs(Integer.valueOf(another[3]))
                        .allowsInplace(Integer.valueOf(another[4]) == 1)
                        .numTArgs(Integer.valueOf(another[5]))
                        .numIArgs(Integer.valueOf(another[6]))
                        .build();

                map.put(another[0], descriptor);
            }

            customOps = Collections.unmodifiableMap(map);
        }

        return customOps;
    }



    protected int[] getShapeFromPointer(IntPointer ptr) {
        val rank = ptr.get(0);
        int[] array = new int[rank];
        for (int i = 0; i < rank; i++) {
            array[i] = ptr.get(i+1);
        }
        return array;
    }

    @Override
    public List<int[]> calculateOutputShape(@NonNull CustomOp op) {

        Nd4j.getExecutioner().commit();

        val lc = op.opName().toLowerCase();
        val hash = op.opHash();

        val result = new ArrayList<int[]>();

        val inputBuffers = new PointerPointer<>(op.inputArguments().length);
        val inputShapes = new PointerPointer<>(op.inputArguments().length);

        int cnt= 0;
        for (val in: op.inputArguments()) {
            // NOT A TYPO: shape functions work on host side only
            inputBuffers.put(cnt, in.data().addressPointer());
            inputShapes.put(cnt++, in.shapeInfoDataBuffer().addressPointer());
        }


        val iArgs = op.iArgs().length > 0 ? new IntPointer(op.iArgs().length) : null;
        cnt = 0;
        for (val i: op.iArgs())
            iArgs.put(cnt++, i);

        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            val tArgs = op.tArgs().length > 0 ? new FloatPointer(op.tArgs().length) : null;

            cnt = 0;
            for (val t: op.tArgs())
                tArgs.put(cnt++, (float) t);

            val ptrptr = (Nd4jCuda.ShapeList) nativeOps.calculateOutputShapesFloat(null, hash, inputBuffers, inputShapes, op.inputArguments().length, tArgs, op.tArgs().length, iArgs, op.iArgs().length);

            if (ptrptr == null)
                throw new RuntimeException();

            for (int e = 0; e < ptrptr.size(); e++ )
                result.add(getShapeFromPointer(new PagedPointer(ptrptr.at(e)).asIntPointer()));

            nativeOps.deleteShapeList(ptrptr);
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            val tArgs = op.tArgs().length > 0 ? new DoublePointer(op.tArgs().length) : null;

            cnt = 0;
            for (val t: op.tArgs())
                tArgs.put(cnt++, (float) t);

            val ptrptr = (Nd4jCuda.ShapeList) nativeOps.calculateOutputShapesDouble(null, hash, inputBuffers, inputShapes, op.inputArguments().length, tArgs, op.tArgs().length, iArgs, op.iArgs().length);

            if (ptrptr == null)
                throw new RuntimeException();

            for (int e = 0; e < ptrptr.size(); e++ )
                result.add(getShapeFromPointer(new PagedPointer(ptrptr.at(e)).asIntPointer()));

            nativeOps.deleteShapeList(ptrptr);
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            val tArgs = op.tArgs().length > 0 ? new ShortPointer(op.tArgs().length) : null;

            cnt = 0;
            for (val t: op.tArgs())
                tArgs.put(cnt++, ArrayUtil.toHalf((float) t));

            val ptrptr = (Nd4jCuda.ShapeList) nativeOps.calculateOutputShapesHalf(null, hash, inputBuffers, inputShapes, op.inputArguments().length, tArgs, op.tArgs().length, iArgs, op.iArgs().length);

            if (ptrptr == null)
                throw new RuntimeException();

            for (int e = 0; e < ptrptr.size(); e++ )
                result.add(getShapeFromPointer(new PagedPointer(ptrptr.at(e)).asIntPointer()));


            nativeOps.deleteShapeList(ptrptr);
        }


        return result;
    }

    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * PLEASE NOTE: right now this operations are executing on CPU
     * @param op
     */
    public void exec(CustomOp op) {

        Nd4j.getExecutioner().commit();

        if (op.opName().equalsIgnoreCase("im2col")) {
            val dtype = Nd4j.dataType();

            val xArr = op.inputArguments()[0];
            val zArr = op.outputArguments()[0];

            CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(zArr, xArr);

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            PointerPointer xShapeHost =
                    extraz.get().put(AddressRetriever.retrieveHostPointer(xArr.shapeInfoDataBuffer()), // 0
                            context.getOldStream(), // 1
                            AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                            context.getBufferAllocation(), // 3
                            context.getBufferReduction(), // 4
                            context.getBufferScalar(), // 5
                            context.getBufferSpecial(),
                            null,
                            AddressRetriever.retrieveHostPointer(zArr.shapeInfoDataBuffer())
                    );


            val x = AtomicAllocator.getInstance().getPointer(xArr, context);
            val z = AtomicAllocator.getInstance().getPointer(zArr, context);

            val xShape = AtomicAllocator.getInstance().getPointer(xArr.shapeInfoDataBuffer(), context);
            val zShape = AtomicAllocator.getInstance().getPointer(zArr.shapeInfoDataBuffer(), context);

            double zeroPad = 0.0;
            if(op.tArgs() != null && op.tArgs().length > 0){
                zeroPad = op.tArgs()[0];
            }
            val extrass = new double[]{op.iArgs()[0], op.iArgs()[1], op.iArgs()[2], op.iArgs()[3], op.iArgs()[4], op.iArgs()[5], op.iArgs()[6], op.iArgs()[7], op.iArgs()[8], zeroPad};
            val extraArgsBuff = Nd4j.getConstantHandler().getConstantBuffer(extrass);
            val extraArgs = AtomicAllocator.getInstance().getPointer(extraArgsBuff, context);


            if (dtype == DataBuffer.Type.DOUBLE) {
                nativeOps.execTransformDouble(xShapeHost, 37, (DoublePointer) x, (IntPointer) xShape, (DoublePointer) z, (IntPointer) zShape, (DoublePointer) extraArgs);
            } else if (dtype == DataBuffer.Type.FLOAT) {
                nativeOps.execTransformFloat(xShapeHost, 37, (FloatPointer) x, (IntPointer) xShape, (FloatPointer) z, (IntPointer) zShape, (FloatPointer) extraArgs);
            } else if (dtype == DataBuffer.Type.HALF) {
                nativeOps.execTransformHalf(xShapeHost, 37, (ShortPointer) x, (IntPointer) xShape, (ShortPointer) z, (IntPointer) zShape, (ShortPointer) extraArgs);
            }

            //AtomicAllocator.getInstance().getAllocationPoint(zArr).tickDeviceWrite();
            AtomicAllocator.getInstance().getFlowController().registerAction(context, zArr, xArr);

            //Nd4j.getExecutioner().commit();

            return;
        } else if (op.opName().equalsIgnoreCase("col2im")) {
            val dtype = Nd4j.dataType();

            val xArr = op.inputArguments()[0];
            val zArr = op.outputArguments()[0];

            CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(zArr, xArr);

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            PointerPointer xShapeHost =
                    extraz.get().put(AddressRetriever.retrieveHostPointer(xArr.shapeInfoDataBuffer()), // 0
                            context.getOldStream(), // 1
                            AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                            context.getBufferAllocation(), // 3
                            context.getBufferReduction(), // 4
                            context.getBufferScalar(), // 5
                            context.getBufferSpecial(),
                            null,
                            AddressRetriever.retrieveHostPointer(zArr.shapeInfoDataBuffer())
                    );


            val x = AtomicAllocator.getInstance().getPointer(xArr, context);
            val z = AtomicAllocator.getInstance().getPointer(zArr, context);

            val xShape = AtomicAllocator.getInstance().getPointer(xArr.shapeInfoDataBuffer(), context);
            val zShape = AtomicAllocator.getInstance().getPointer(zArr.shapeInfoDataBuffer(), context);

            val extrass = new double[]{op.iArgs()[0], op.iArgs()[1], op.iArgs()[2], op.iArgs()[3], op.iArgs()[4], op.iArgs()[5], op.iArgs()[6], op.iArgs()[7]};
            val extraArgsBuff = Nd4j.getConstantHandler().getConstantBuffer(extrass);
            val extraArgs = AtomicAllocator.getInstance().getPointer(extraArgsBuff, context);

            if (dtype == DataBuffer.Type.DOUBLE) {
                nativeOps.execTransformDouble(xShapeHost, 36, (DoublePointer) x, (IntPointer) xShape, (DoublePointer) z, (IntPointer) zShape, (DoublePointer) extraArgs);
            } else if (dtype == DataBuffer.Type.FLOAT) {
                nativeOps.execTransformFloat(xShapeHost, 36, (FloatPointer) x, (IntPointer) xShape, (FloatPointer) z, (IntPointer) zShape, (FloatPointer) extraArgs);
            } else if (dtype == DataBuffer.Type.HALF) {
                nativeOps.execTransformHalf(xShapeHost, 36, (ShortPointer) x, (IntPointer) xShape, (ShortPointer) z, (IntPointer) zShape, (ShortPointer) extraArgs);
            }

            //AtomicAllocator.getInstance().getAllocationPoint(zArr).tickDeviceWrite();
            AtomicAllocator.getInstance().getFlowController().registerAction(context, zArr, xArr);

            //Nd4j.getExecutioner().commit();

            return;
        } else if (op.opName().equalsIgnoreCase("pooling2d")) {
            val dtype = Nd4j.dataType();

            val xArr = op.inputArguments()[0];
            val zArr = op.outputArguments()[0];

            CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(zArr, xArr);

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            PointerPointer xShapeHost =
                    extraz.get().put(AddressRetriever.retrieveHostPointer(xArr.shapeInfoDataBuffer()), // 0
                            context.getOldStream(), // 1
                            AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                            context.getBufferAllocation(), // 3
                            context.getBufferReduction(), // 4
                            context.getBufferScalar(), // 5
                            context.getBufferSpecial(),
                            null,
                            AddressRetriever.retrieveHostPointer(zArr.shapeInfoDataBuffer())
                    );


            val x = AtomicAllocator.getInstance().getPointer(xArr, context);
            val z = AtomicAllocator.getInstance().getPointer(zArr, context);

            val xShape = AtomicAllocator.getInstance().getPointer(xArr.shapeInfoDataBuffer(), context);
            val zShape = AtomicAllocator.getInstance().getPointer(zArr.shapeInfoDataBuffer(), context);

            val extrass = new double[]{op.iArgs()[0], op.iArgs()[1], op.iArgs()[2], op.iArgs()[3], op.iArgs()[4], op.iArgs()[5], op.iArgs()[6], op.iArgs()[7], op.iArgs()[8]};
            val extraArgsBuff = Nd4j.getConstantHandler().getConstantBuffer(extrass);
            val extraArgs = AtomicAllocator.getInstance().getPointer(extraArgsBuff, context);

            if (dtype == DataBuffer.Type.DOUBLE) {
                nativeOps.execTransformDouble(xShapeHost, 71, (DoublePointer) x, (IntPointer) xShape, (DoublePointer) z, (IntPointer) zShape, (DoublePointer) extraArgs);
            } else if (dtype == DataBuffer.Type.FLOAT) {
                nativeOps.execTransformFloat(xShapeHost, 71, (FloatPointer) x, (IntPointer) xShape, (FloatPointer) z, (IntPointer) zShape, (FloatPointer) extraArgs);
            } else if (dtype == DataBuffer.Type.HALF) {
                nativeOps.execTransformHalf(xShapeHost, 71, (ShortPointer) x, (IntPointer) xShape, (ShortPointer) z, (IntPointer) zShape, (ShortPointer) extraArgs);
            }

            // AtomicAllocator.getInstance().getAllocationPoint(zArr).tickDeviceWrite();
            AtomicAllocator.getInstance().getFlowController().registerAction(context, zArr, xArr);

            //Nd4j.getExecutioner().commit();

            return;
        }

        Nd4j.getExecutioner().commit();

        long st = profilingHookIn(op);

        CudaContext context =(CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        //AtomicAllocator.getInstance().getFlowController().prepareActionAllWrite(op.outputArguments());

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));


        PointerPointer extras = extraz.get().put(
                new CudaPointer(1),
                context.getOldStream(),
                context.getBufferScalar(),
                context.getBufferReduction());

        val outputArgs = op.outputArguments();
        val inputArgs = op.inputArguments();

        if (outputArgs.length == 0 && !op.isInplaceCall())
            throw new ND4JIllegalStateException("You can't execute non-inplace CustomOp without outputs being specified");

        val lc = op.opName().toLowerCase();
        val hash = op.opHash();


        val inputShapes = new PointerPointer<>(inputArgs.length * 2);
        val inputBuffers = new PointerPointer<>(inputArgs.length * 2);

        int cnt= 0;
        for (val in: inputArgs) {
            val hp = AtomicAllocator.getInstance().getHostPointer(in.shapeInfoDataBuffer());
            inputBuffers.put(cnt,  AtomicAllocator.getInstance().getHostPointer(in));
            inputShapes.put(cnt, hp);


            val dp = AtomicAllocator.getInstance().getPointer(in.shapeInfoDataBuffer(), context);

            inputBuffers.put(cnt + inputArgs.length, AtomicAllocator.getInstance().getPointer(in, context));
            inputShapes.put(cnt+ inputArgs.length, dp);

            if (op.isInplaceCall())
                AtomicAllocator.getInstance().getAllocationPoint(in).tickHostWrite();

            cnt++;
        }


        val outputShapes = new PointerPointer<>(outputArgs.length * 2);
        val outputBuffers = new PointerPointer<>(outputArgs.length * 2);

        cnt= 0;
        for (val out: outputArgs) {
            outputBuffers.put(cnt,  AtomicAllocator.getInstance().getHostPointer(out));
            outputShapes.put(cnt,  AtomicAllocator.getInstance().getHostPointer(out.shapeInfoDataBuffer()));

            outputBuffers.put(cnt + outputArgs.length,  AtomicAllocator.getInstance().getPointer(out, context));
            outputShapes.put(cnt + outputArgs.length,  AtomicAllocator.getInstance().getPointer(out.shapeInfoDataBuffer(), context));

            AtomicAllocator.getInstance().getAllocationPoint(out).tickHostWrite();

            cnt++;
        }

        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            val tArgs = op.tArgs().length > 0 ? new FloatPointer(op.tArgs().length) : null;
            val iArgs = op.iArgs().length > 0 ? new IntPointer(op.iArgs().length) : null;

            cnt = 0;
            for (val t: op.tArgs())
                tArgs.put(cnt++, (float) t);

            cnt = 0;
            for (val i: op.iArgs())
                iArgs.put(cnt++, i);

            val status = OpStatus.byNumber(nativeOps.execCustomOpFloat(extras, hash, inputBuffers, inputShapes, inputArgs.length, outputBuffers, outputShapes, outputArgs.length, tArgs, op.tArgs().length, iArgs, op.iArgs().length, op.isInplaceCall()));
            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            val tArgs = op.tArgs().length > 0 ? new DoublePointer(op.tArgs().length) : null;
            val iArgs = op.iArgs().length > 0 ? new IntPointer(op.iArgs().length) : null;

            cnt = 0;
            for (val t: op.tArgs())
                tArgs.put(cnt++, t);

            for (val i: op.iArgs())
                iArgs.put(cnt++, i);

            val status = OpStatus.byNumber(nativeOps.execCustomOpDouble(extras, hash, inputBuffers, inputShapes, inputArgs.length, outputBuffers, outputShapes, outputArgs.length, tArgs, op.tArgs().length, iArgs, op.iArgs().length, op.isInplaceCall()));
            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            val tArgs = op.tArgs().length > 0 ? new ShortPointer(op.tArgs().length) : null;
            val iArgs = op.iArgs().length > 0 ? new IntPointer(op.iArgs().length) : null;

            cnt = 0;
            for (val t: op.tArgs())
                tArgs.put(cnt++, ArrayUtil.toHalf((float) t));

            cnt = 0;
            for (val i: op.iArgs())
                iArgs.put(cnt++, i);

            val status = OpStatus.byNumber(nativeOps.execCustomOpHalf(extras, hash, inputBuffers, inputShapes, inputArgs.length, outputBuffers, outputShapes, outputArgs.length, tArgs, op.tArgs().length, iArgs, op.iArgs().length, op.isInplaceCall()));
            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);
        }

        //AtomicAllocator.getInstance().getFlowController().prepareActionAllWrite(op.outputArguments());

        profilingHookOut(op, st);
    }

    @Override
    public void enableDebugMode(boolean reallyEnable) {
        nativeOps.enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        nativeOps.enableVerboseMode(reallyEnable);
    }

    @Override
    public void registerGraph(long id, Pointer graph) {
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT)
            nativeOps.registerGraphFloat(null, id, graph);
        else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            nativeOps.registerGraphDouble(null, id, graph);
        else if (Nd4j.dataType() == DataBuffer.Type.HALF)
            nativeOps.registerGraphHalf(null, id, graph);
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, Map<String, INDArray> map) {

        this.commit();

        val ptrBuffers = new PointerPointer(map.size() * 2);
        val ptrShapes = new PointerPointer(map.size() * 2);
        val ptrIndices = new IntPointer(map.size());

        int cnt = 0;
        val keySet = new ArrayList<String>(map.keySet());
        for (val key: keySet) {
            val array = map.get(key);

            ptrBuffers.put(cnt, AtomicAllocator.getInstance().getHostPointer(array));
            ptrShapes.put(cnt, AtomicAllocator.getInstance().getHostPointer(array.shapeInfoDataBuffer()));
            ptrIndices.put(cnt, cnt);

            cnt++;
        }

        val newMap = new LinkedHashMap<String, INDArray>();
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            val result = (Nd4jCuda.FloatVariablesSet) nativeOps.executeStoredGraphFloat(null, id, ptrBuffers, ptrShapes, ptrIndices, map.size());

            val status = OpStatus.byNumber(result.status());

            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);

            for (int e = 0; e < result.size(); e++) {
                val var = result.at(e);
                val nodeId = var.id();
                val index = var.index();
                val shapeInfo = var.getNDArray().shapeInfo();
                val buffer = var.getNDArray().buffer();

                val rank = shapeInfo.get(0);
                val jshape = new int[rank * 2 + 4];
                for (int i = 0; i < jshape.length; i++) {
                    jshape[i] = shapeInfo.get(i);
                }

                val shapeOf = Shape.shapeOf(jshape);
                val stridesOf = Shape.stridesOf(jshape);
                val order = Shape.order(jshape);
                val array = Nd4j.create(shapeOf, stridesOf, 0, order);


                Pointer.memcpy(AtomicAllocator.getInstance().getHostPointer(array), buffer, ArrayUtil.prod(shapeOf) * Nd4j.sizeOfDataType());
                AtomicAllocator.getInstance().getAllocationPoint(array).tickHostWrite();

                newMap.put(keySet.get(nodeId), array);
            }
            nativeOps.deleteVariablesSetFloat(result);
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            val result = (Nd4jCuda.DoubleVariablesSet) nativeOps.executeStoredGraphDouble(null, id, ptrBuffers, ptrShapes, ptrIndices, map.size());

            val status = OpStatus.byNumber(result.status());

            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);

            for (int e = 0; e < result.size(); e++) {
                val var = result.at(e);
                val nodeId = var.id();
                val index = var.index();
                val shapeInfo = var.getNDArray().shapeInfo();
                val buffer = var.getNDArray().buffer();

                val rank = shapeInfo.get(0);
                val jshape = new int[rank * 2 + 4];
                for (int i = 0; i < jshape.length; i++) {
                    jshape[i] = shapeInfo.get(i);
                }

                val shapeOf = Shape.shapeOf(jshape);
                val stridesOf = Shape.stridesOf(jshape);
                val order = Shape.order(jshape);
                val array = Nd4j.create(shapeOf, stridesOf, 0, order);


                Pointer.memcpy(AtomicAllocator.getInstance().getHostPointer(array), buffer, ArrayUtil.prod(shapeOf) * Nd4j.sizeOfDataType());
                AtomicAllocator.getInstance().getAllocationPoint(array).tickHostWrite();

                newMap.put(keySet.get(nodeId), array);
            }

            nativeOps.deleteVariablesSetDouble(result);
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            val result = (Nd4jCuda.DoubleVariablesSet) nativeOps.executeStoredGraphHalf(null, id, ptrBuffers, ptrShapes, ptrIndices, map.size());

            val status = OpStatus.byNumber(result.status());

            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);

            for (int e = 0; e < result.size(); e++) {
                val var = result.at(e);
                val nodeId = var.id();
                val index = var.index();
                val shapeInfo = var.getNDArray().shapeInfo();
                val buffer = var.getNDArray().buffer();

                val rank = shapeInfo.get(0);
                val jshape = new int[rank * 2 + 4];
                for (int i = 0; i < jshape.length; i++) {
                    jshape[i] = shapeInfo.get(i);
                }

                val shapeOf = Shape.shapeOf(jshape);
                val stridesOf = Shape.stridesOf(jshape);
                val order = Shape.order(jshape);
                val array = Nd4j.create(shapeOf, stridesOf, 0, order);


                Pointer.memcpy(AtomicAllocator.getInstance().getHostPointer(array), buffer, ArrayUtil.prod(shapeOf) * Nd4j.sizeOfDataType());
                AtomicAllocator.getInstance().getAllocationPoint(array).tickHostWrite();

                newMap.put(keySet.get(nodeId), array);
            }

            nativeOps.deleteVariablesSetHalf(result);
        }

        return newMap;
    }

    @Override
    public void forgetGraph(long id) {
        nativeOps.unregisterGraph(null, id);
    }

    /**
     * This method allows to set desired number of elements per thread, for performance optimization purposes.
     * I.e. if array contains 2048 elements, and threshold is set to 1024, 2 threads will be used for given op execution.
     * <p>
     * Default value: 1024
     *
     * @param threshold
     */
    @Override
    public void setElementsThreshold(int threshold) {
        nativeOps.setElementThreshold(threshold);
    }

    /**
     * This method allows to set desired number of sub-arrays per thread, for performance optimization purposes.
     * I.e. if matrix has shape of 64 x 128, and threshold is set to 8, each thread will be processing 8 sub-arrays (sure, if you have 8 core cpu).
     * If your cpu has, say, 4, cores, only 4 threads will be spawned, and each will process 16 sub-arrays
     * <p>
     * Default value: 8
     *
     * @param threshold
     */
    @Override
    public void setTadThreshold(int threshold) {
        nativeOps.setTADThreshold(threshold);
    }
}


