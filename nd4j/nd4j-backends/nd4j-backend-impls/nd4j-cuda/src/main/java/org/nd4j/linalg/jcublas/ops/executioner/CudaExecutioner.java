/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.jcublas.ops.executioner;


import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.tad.DeviceTADManager;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.bindings.Nd4jCuda;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaLongDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaUtf8Buffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.*;

import java.util.*;

import static org.bytedeco.cuda.global.cudart.*;


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


    @Getter
    protected static TADManager tadManager = new DeviceTADManager();
    protected ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();
    protected volatile transient Properties properties;

    protected ThreadLocal<String> lastOp = new ThreadLocal<>();

    protected Map<String, CustomOpDescriptor> customOps = null;

    protected AtomicBoolean experimentalMode = new AtomicBoolean(false);

    public CudaExecutioner() {
        experimentalMode.set(Nd4j.getNativeOps().isExperimentalEnabled());
    }


    @Override
    public String getLastOp() {
        return lastOp.get();
    }

    @Override
    public INDArray exec(BroadcastOp op) {
        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);

        INDArray dim1 = op.dimensions().castTo(DataType.LONG);
        val dimension = OpaqueNDArray.fromINDArray(dim1);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        Pointer hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        val x = OpaqueNDArray.fromINDArray(op.x());
        val y = OpaqueNDArray.fromINDArray(op.y());
        val z = OpaqueNDArray.fromINDArray(op.z());

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(),dim1.toLongVector());

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dim1.toLongVector());

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                devTadShapeInfoZ, devTadOffsetsZ);
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(x.dataType()), context) : null;

        switch (op.getOpType()) {
            case BROADCAST:
                Nd4j.getNativeOps().execBroadcast(xShapeInfoHostPointer, op.opNum(),
                        x,
                        y,
                        z,
                        extraArgs,
                        dimension);
                break;
            case BROADCAST_BOOL:
                Nd4j.getNativeOps().execBroadcastBool(xShapeInfoHostPointer, op.opNum(),
                        x,
                        y,
                        z,
                        extraArgs,
                        dimension);
                break;
            default:
                throw new UnsupportedOperationException("Unknown op type: " + op.getOpType());
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, null, st);

        return op.z();
    }

    /**
     *
     * @param op
     * @param dimension
     * @return
     */
    protected INDArray naiveExec(ReduceOp op, long... dimension) {
        long st = profilingConfigurableHookIn(op);

        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()) {
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(op.z() != null){
                Preconditions.checkState(op.x().equalShapes(op.z()), "For empty reductions, result (z) array must have same shape as x shape." +
                        " Got: x=%ndShape, z=%ndShape", op.x(), op.z());
                op.z().assign(op.x());
                return op.z();
            } else {
                op.setZ(op.x().dup());
                return op.z();
            }
        }

        INDArray ret = op.z();

        checkForCompression(op);
        op.validateDataTypes(null);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val hostXShapeInfo = op.x() == null ? null : AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer());
        val hostYShapeInfo = op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        val hostZShapeInfo = op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

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
            if (dimension.length == 0 || (dimension.length == 1 &&  dimension[0] == Integer.MAX_VALUE )|| op.x().tensorAlongDimension(0, dimension).length() != op.y().length()) {
                if (!op.isComplexAccumulation() && op.x().length() != op.y().length())
                    throw new ND4JIllegalStateException("Op.X [" + op.x().length() + "] and Op.Y [" + op.y().length() + "] lengths should match");

                if (!op.z().isScalar()) {
                    Pair<DataBuffer, DataBuffer> yTadBuffers = tadManager.getTADOnlyShapeInfo(op.y(), dimension);

                    yDevTadShapeInfo = AtomicAllocator.getInstance().getPointer(yTadBuffers.getFirst(), context);

                    DataBuffer yOffsets = yTadBuffers.getSecond();
                    yDevTadOffsets = yOffsets == null ? null : AtomicAllocator.getInstance().getPointer(yOffsets, context);

                    xShapeInfoHostPointer.put(12, yDevTadShapeInfo);
                    xShapeInfoHostPointer.put(13, yDevTadOffsets);
                }
            } else {
                // TAD vs full array code branch
                val fakeOffsets = Nd4j.getConstantHandler().getConstantBuffer(new int[] {0, 0}, DataType.LONG);
                yDevTadOffsets = fakeOffsets == null ? null : AtomicAllocator.getInstance().getPointer(fakeOffsets, context);

                yDevTadShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);

                xShapeInfoHostPointer.put(12, AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context));
                xShapeInfoHostPointer.put(13, null);
            }
        }

        DataType argsType;
        switch (op.getOpType()) {
            case REDUCE_LONG:
            case REDUCE_BOOL:
                argsType = op.x().dataType();
                break;
            default:
                argsType = op.z().dataType();
        }

        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(argsType), context) : null;
        Pointer dimensionPointer = AtomicAllocator.getInstance().getPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension), context); //AtomicAllocator.getInstance().getPointer(Nd4j.createBuffer(dimension), context);

        val x = OpaqueNDArray.fromINDArray(op.x());
        val y = OpaqueNDArray.fromINDArray(op.y());
        val z = OpaqueNDArray.fromINDArray(op.z());
        INDArray dimArr = Nd4j.createFromArray(dimension);
        val dim = OpaqueNDArray.fromINDArray(dimArr);
        if (op instanceof Variance) {
            if (ret.isScalar()) {
                Nd4j.getNativeOps().execSummaryStatsScalar(xShapeInfoHostPointer, op.opNum(),
                        x,
                        extraArgs,
                        z,
                        ((Variance) op).isBiasCorrected());
            } else {
                Nd4j.getNativeOps().execSummaryStatsTad(xShapeInfoHostPointer, op.opNum(),
                        x,
                        extraArgs,
                        z,
                        dim,
                        ((Variance) op).isBiasCorrected());
            }
        } else if (op.y() != null) {
            if (ret.isScalar()) {
                Nd4j.getNativeOps().execReduce3Scalar(xShapeInfoHostPointer, op.opNum(),
                        x,
                        extraArgs,
                        y,
                        z);
            } else {
                Nd4j.getNativeOps().execReduce3Tad(xShapeInfoHostPointer, op.opNum(),x
                        ,extraArgs,y,z,dim);
            }
        } else {
            if (ret.isScalar()) {
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        Nd4j.getNativeOps().execReduceFloat(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,
                                z);
                        break;
                    case REDUCE_BOOL:
                        Nd4j.getNativeOps().execReduceBool(xShapeInfoHostPointer,
                                op.opNum(),
                                x,
                                extraArgs,
                                z,dim);
                        break;
                    case REDUCE_LONG:
                        Nd4j.getNativeOps().execReduceLong(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,
                                z,dim);
                        break;
                    case REDUCE_SAME:
                        Nd4j.getNativeOps().execReduceSame(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,
                                z);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
            } else {
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        Nd4j.getNativeOps().execReduceFloat2(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,z,dim);
                        break;
                    case REDUCE_BOOL:
                        Nd4j.getNativeOps().execReduceBool2(xShapeInfoHostPointer, op.opNum(),
                                x,extraArgs,z,dim);
                        break;
                    case REDUCE_SAME:
                        Nd4j.getNativeOps().execReduceSame2(xShapeInfoHostPointer, op.opNum(),
                                x, extraArgs, z,dim);
                        break;
                    case REDUCE_LONG:
                        Nd4j.getNativeOps().execReduceLong2(xShapeInfoHostPointer, op.opNum(),
                                x, extraArgs, z,dim);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
            }
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, null, st);

        return op.z();
    }

    @Override
    public INDArray exec(Variance op) {
        return exec((ReduceOp) op);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        checkForCompression(op);

        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()){
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(op.z() != null) {
                Preconditions.checkState(op.x().equalShapes(op.z()), "For empty reductions, result (z) array must have same shape as x shape." +
                        " Got: x=%ndShape, z=%ndShape", op.x(), op.z());
                op.z().assign(op.x());
                return op.z();
            } else {
                op.setZ(op.x().dup());
                return op.z();
            }
        }

        val dimension = op.dimensions().toLongVector();

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val maxShape = Shape.getMaxShape(op.x(),op.y());

        val wholeDims = Shape.wholeArrayDimension(dimension) || op.x().rank() == dimension.length || dimension.length == 0;
        val retShape = Shape.reductionShape(op.y() == null ? op.x() : op.x().length() > op.y().length() ? op.x() : op.y(), dimension, true, op.isKeepDims());

        if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape) && ArrayUtil.prodLong(retShape) > 1 && op.y() == null)
            return op.noOp();

        val dtype = op.resultType();
        INDArray ret = null;
        if (op.z() == null || op.z() == op.x()) {
            if (op.isComplexAccumulation()) {
                val xT = op.x().tensorsAlongDimension(dimension);
                val yT = op.y().tensorsAlongDimension(dimension);

                // we intentionally want to set it to 0.0
                ret = Nd4j.createUninitialized(dtype, new long[] {xT, yT});
            } else {
                if (op.y() != null) {
                    //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
                    if (op.x().length() == op.y().length()) {
                        //Pairwise
                        if (!wholeDims && op.x().tensorsAlongDimension(dimension) != op.y().tensorsAlongDimension(dimension)) {
                            throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                                    Arrays.toString(op.x().shape()) + ", y shape = " + Arrays.toString(op.y().shape()) +
                                    ", dimension = " + Arrays.toString(dimension) + ")");
                        }
                    } else {
                        if (dimension.length == 0)
                            throw new ND4JIllegalStateException("TAD vs TAD comparison requires dimension (or other comparison mode was supposed to be used?)");

                        //Every X TAD vs. entirety of Y
                        val xTADSize = op.x().length() / op.x().tensorsAlongDimension(dimension);

                        if (xTADSize != op.y().length()) {
                            throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                                    " (x TAD size = " + xTADSize + ", y size = " + op.y().length());
                        }
                    }
                }

                // in case of regular accumulation we don't care about array state before op
                ret = Nd4j.create(dtype, retShape);
            }
            op.setZ(ret);
        } else {
            // compare length

            if (op.z().length() != (retShape.length == 0 ? 1 : ArrayUtil.prodLong(retShape)))
                throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + Arrays.toString(retShape) + "]");
        }

        long st = profilingConfigurableHookIn(op);
        naiveExec(op, dimension);

        profilingConfigurableHookOut(op, null, st);

        return op.z();
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        val dimension = Shape.normalizeAxis(op.x().rank(), op.dimensions().toLongVector());

        if (op.x().isEmpty()) {
            for (val d:dimension) {
                Preconditions.checkArgument(op.x().size(d) != 0, "IndexReduce can't be issued along axis with 0 in shape");
            }
        }

        if (op.z() == null) {
            val retShape = Shape.reductionShape(op.x(), dimension, true, op.isKeepDims());
            op.setZ(Nd4j.createUninitialized(DataType.LONG, retShape));
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (op.x().isVector() && op.x().length() == op.z().length()) {
            return op.x();
        }

        if (op.z().isEmpty())
            return op.z();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val context = AtomicAllocator.getInstance().getDeviceContext();

        val hostXShapeInfo =
                op.x() == null ? null : AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer());
        val hostYShapeInfo =
                op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        val hostZShapeInfo =
                op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        val xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);
        val zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = tadBuffers.getSecond();
        val devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);
        Pointer extraArgs = op.extraArgs() != null
                ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(op.x().dataType()), context) : null;

        val x = OpaqueNDArray.fromINDArray(op.x());
        val y = OpaqueNDArray.fromINDArray(op.y());
        val z = OpaqueNDArray.fromINDArray(op.z());
        INDArray dim1 = op.dimensions().castTo(DataType.LONG);
        val dimension2 = OpaqueNDArray.fromINDArray(dim1);
        Nd4j.getNativeOps().execIndexReduce(xShapeInfoHostPointer, op.opNum(),
                x,
                extraArgs,
                z,
                dimension2);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, null, st);

        return op.z();
    }


    @Override
    public INDArray exec(Op op) {
        return exec(op, null);
    }

    @Override
    public INDArray exec(Op op, OpContext oc) {
        checkForCompression(op);

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t, oc);
        } else if (op instanceof ReduceOp) {
            ReduceOp acc = (ReduceOp) op;
            invoke(acc, oc, acc.dimensionsArr());
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc, oc);
        } else if (op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            invoke(broadcastOp, oc);
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) op;
            invoke(indexAccumulation, oc, indexAccumulation.dimensions().toLongVector());
        } else if (op instanceof RandomOp) {
            exec((RandomOp) op, oc, Nd4j.getRandom());
        } else if (op instanceof CustomOp) {
            exec((CustomOp) op, oc);
        }


        return op.z();
    }


    @Override
    public TransformOp execAndReturn(TransformOp op) {
        checkForCompression(op);
        invoke(op, null);
        return op;
    }



    protected CudaContext invoke(BroadcastOp op, OpContext oc) {
        long st = profilingConfigurableHookIn(op);

        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        checkForCompression(op);


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(x.shapeInfoDataBuffer(), context);


        val hostXShapeInfo =
                x == null ? null : AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer());
        val hostYShapeInfo =
                y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());
        val hostZShapeInfo =
                z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        val tadBuffers = tadManager.getTADOnlyShapeInfo(x, op.getDimension());

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = tadBuffers.getSecond();
        val devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        // that's the place where we're going to have second TAD in place
        val tadBuffersZ = tadManager.getTADOnlyShapeInfo(z, op.getDimension());

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), // 0
                context.getOldStream(), // 1
                AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                context.getBufferAllocation(), // 3
                context.getBufferReduction(),  // 4
                context.getBufferScalar(),  // 5
                context.getBufferSpecial(), // 6
                hostYShapeInfo,  // 7
                hostZShapeInfo,  // 8
                hostTadShapeInfo,  // 9
                devTadShapeInfo,  // 10
                devTadOffsets, // 11
                devTadShapeInfoZ,  // 12
                devTadOffsetsZ); // 13

        Pointer yShapeInfo = AtomicAllocator.getInstance().getPointer(y.shapeInfoDataBuffer(), context);

        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(z.shapeInfoDataBuffer(), context);
        Pointer dimensionPointer = AtomicAllocator.getInstance().getPointer(AtomicAllocator.getInstance().getConstantBuffer(op.getDimension()), context);

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);
        val dimension = OpaqueNDArray.fromINDArray(op.dimensions().castTo(DataType.LONG));
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(x.dataType()), context) : null;

        switch (op.getOpType()) {
            case BROADCAST:
                Nd4j.getNativeOps().execBroadcast(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        yb,
                        zb,
                        extraArgs,
                        dimension);
                break;
            case BROADCAST_BOOL:
                Nd4j.getNativeOps().execBroadcastBool(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        yb,
                        zb,
                        extraArgs,
                        dimension);
                break;
            default:
                throw new UnsupportedOperationException("Unknown opType: " + op.getOpType());
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, oc, st);

        return null;
    }



    protected CudaContext invoke(IndexAccumulation op, OpContext oc, long[] dimension) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        dimension = Shape.normalizeAxis(x.rank(), dimension);
        if (dimension == null || (dimension.length == 1 && dimension[0] == Integer.MAX_VALUE)) {
            if(z == x || z == null) {
                z = Nd4j.createUninitialized(DataType.LONG, new long[0], 'c');
                setZ(z, op, oc);
            }
        }

        boolean keepDims = op.isKeepDims();
        long[] retShape = Shape.reductionShape(x, dimension, true, keepDims);

        if(z == null || x == z) {
            val ret = Nd4j.createUninitialized(DataType.LONG, retShape);

            setZ(ret, op, oc);
            z = ret;
        } else if(!Arrays.equals(retShape, z.shape())){
            throw new IllegalStateException("Z array shape does not match expected return type for op " + op
                    + ": expected shape " + Arrays.toString(retShape) + ", z.shape()=" + Arrays.toString(z.shape()));
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());
        CudaEnvironment.getInstance().getConfiguration().enableDebug(true);
        if (dimension != null)
            for (int i = 0; i < dimension.length; i++)
                if (dimension[i] >= x.rank() && dimension[i] != Integer.MAX_VALUE)
                    throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension) + " contains element that higher then rank of op.X: [" + x.rank() + "]");

        val context = AtomicAllocator.getInstance().getDeviceContext();

        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(x.shapeInfoDataBuffer(), context);
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(x.dataType()), context) : null;

        val hostXShapeInfo = x == null ? null : AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer());
        val hostYShapeInfo = y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        long fdimension[] = dimension;
        if (fdimension == null)
            fdimension = new long[] {0};

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(x, fdimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        val xb = OpaqueNDArray.fromINDArray(x);
        val zb = OpaqueNDArray.fromINDArray(z);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);

        if (z.isScalar() || dimension == null || dimension[0] == Integer.MAX_VALUE) {
            Nd4j.getNativeOps().execIndexReduceScalar(xShapeInfoHostPointer, op.opNum(),
                    xb,
                    extraArgs,
                    zb);
        } else {
            if (dimension != null && dimension.length > 1)
                Arrays.sort(dimension);

            val dim = Nd4j.createFromArray(dimension);
            val dimPointer = OpaqueNDArray.fromINDArray(dim);

            Nd4j.getNativeOps().execIndexReduce(xShapeInfoHostPointer, op.opNum(),
                    xb,
                    extraArgs,
                    zb,
                    dimPointer);
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, oc, st);

        return null;

    }


    protected CudaContext invoke(ReduceOp op, OpContext oc, long[] dimension) {
        val context = AtomicAllocator.getInstance().getDeviceContext();

        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()) {
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(z != null) {
                if(!x.isScalar() && !z.isScalar())
                    Preconditions.checkState(x.equalShapes(z), "For empty reductions, result (z) array must have same shape as x shape." +
                            " Got: x=%ndShape, z=%ndShape", x, z);
                z.assign(x);
                return context;
            } else {
                setZ(x.dup(), op, oc);
                return context;
            }
        }

        // FIXME: this should be moved down to C++ on per-op basis
        // reduce to scalar case, ReduceBool ops require special treatment
        if (op instanceof BaseReduceBoolOp && x.isEmpty() && (dimension == null || (dimension.length == 1 && dimension[0] == Integer.MAX_VALUE))) {
            if (z == null) {
                op.setZ(Nd4j.scalar(((BaseReduceBoolOp) op).emptyValue()));
            } else {
                z.assign(((BaseReduceBoolOp) op).emptyValue());
            }

            return context;
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);

        dimension = Shape.normalizeAxis(x.rank(), dimension);


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        // dimension is ALWAYS null here.
        if (dimension == null )
            dimension = new long[] {Integer.MAX_VALUE};

        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= x.rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + x.rank() + "]");

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val tadBuffers = x.isEmpty() ? Pair.<DataBuffer, DataBuffer>makePair(x.data(), null) : tadManager.getTADOnlyShapeInfo(x, dimension);

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = x.isEmpty() ? null : tadBuffers.getSecond();
        val devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer((DataBuffer) offsets, context);

        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(x.shapeInfoDataBuffer(), context);

        long[] retShape = Shape.reductionShape(x, dimension, true, op.isKeepDims());

        if (y != null) {
            //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
            if (x.length() == y.length()) {
                //Pairwise
                if (x.tensorsAlongDimension(dimension) != y.tensorsAlongDimension(dimension)) {
                    throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                            Arrays.toString(x.shape()) + ", y shape = " + Arrays.toString(y.shape()) +
                            ", dimension = " + Arrays.toString(dimension) + ")");
                }
            } else if(!(op instanceof ReduceOp)) {
                //Every X TAD vs. entirety of Y
                val xTADSize = x.length() / x.tensorsAlongDimension(dimension);

                if (xTADSize != y.length()) {
                    throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                            " (x TAD size = " + xTADSize + ", y size = " + y.length());
                }
            }
        }



        val dataType = oc != null ? op.resultType(oc) : op.resultType();

        if( z == null ) {
            val ret = Nd4j.createUninitialized(dataType, retShape);
            setZ(ret, op, oc);
            z = ret;
        } else if(z.dataType() != dataType || !Arrays.equals(retShape, z.shape())){
            throw new ND4JIllegalStateException("Output array for op " + op.getClass().getSimpleName() + " should have type " + dataType + " and shape " + Arrays.toString(retShape)
                    + " but has datatype " + z.dataType() + " and shape " + Arrays.toString(z.shape()));
        }

        val eb = op.extraArgsDataBuff(z.dataType() == DataType.BOOL || op.getOpType() == Op.Type.REDUCE_LONG ? x.dataType() : z.dataType());
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(eb, context) : null;

        val hostXShapeInfo = x == null ? null : AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer());
        val hostYShapeInfo = y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        val xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);

        val yTadBuffers = y == null ? null : tadManager.getTADOnlyShapeInfo(y, dimension);

        val yDevTadShapeInfo = y == null ? null : AtomicAllocator.getInstance().getPointer(yTadBuffers.getFirst(), context);
        val yOffsets = y == null ? null : yTadBuffers.getSecond();
        val yDevTadOffsets = yOffsets == null ? null : AtomicAllocator.getInstance().getPointer(yOffsets, context);

        if (y != null) {
            xShapeInfoHostPointer.put(12L, yDevTadShapeInfo);
            xShapeInfoHostPointer.put(13L, yDevTadOffsets);
        }

        val zShapeInfo = AtomicAllocator.getInstance().getPointer(z.shapeInfoDataBuffer(), context);

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);
        INDArray dim = Nd4j.createFromArray(dimension);
        OpaqueNDArray dimensionPointer = OpaqueNDArray.fromINDArray(dim);
        op.validateDataTypes(null);

        if (z.isScalar()) {
            if (op instanceof Variance) {
                Nd4j.getNativeOps().execSummaryStatsScalar(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        extraArgs,
                        zb,
                        ((Variance) op).isBiasCorrected());
            } else if (y != null) {
                Nd4j.getNativeOps().execReduce3Scalar(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        extraArgs,
                        yb,
                        zb);
            } else {
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        Nd4j.getNativeOps().execReduceFloat(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb);
                        break;
                    case REDUCE_BOOL:
                        Nd4j.getNativeOps().execReduceBool(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb,dimensionPointer);
                        break;
                    case REDUCE_SAME:
                        Nd4j.getNativeOps().execReduceSame(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb);
                        break;
                    case REDUCE_LONG:
                        Nd4j.getNativeOps().execReduceLong(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb,dimensionPointer);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
            }
        } else {

            if (y != null) {
                Nd4j.getNativeOps().execReduce3Tad(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        xb,
                        extraArgs,
                        yb,
                        zb,dimensionPointer);
            } else {
                if (op instanceof Variance) {
                    Nd4j.getNativeOps().execSummaryStatsTad(xShapeInfoHostPointer, op.opNum(),
                            xb, (LongPointer)
                                    extraArgs,
                            zb,
                            dimensionPointer,
                            ((Variance) op).isBiasCorrected());
                } else {
                    switch (op.getOpType()) {
                        case REDUCE_FLOAT:
                            Nd4j.getNativeOps().execReduceFloat2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);

                            break;
                        case REDUCE_SAME:
                            Nd4j.getNativeOps().execReduceSame2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);
                            break;
                        case REDUCE_BOOL:
                            Nd4j.getNativeOps().execReduceBool2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);
                            break;
                        case REDUCE_LONG:
                            Nd4j.getNativeOps().execReduceLong2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);
                            break;
                        default:
                            throw new UnsupportedOperationException();
                    }
                }
            }
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, oc, st);

        Nd4j.getExecutioner().commit();

        return context;
    }


    protected CudaContext intercept(ScalarOp op, long[] dimension) {
        long st = profilingConfigurableHookIn(op);

        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val hostXShapeInfo = op.x() == null ? null : AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer());
        val hostYShapeInfo = op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        val hostZShapeInfo = op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        val xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);
        val yShapeInfo = AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context);
        val zShapeInfo = AtomicAllocator.getInstance().getPointer(op.z().shapeInfoDataBuffer(), context);

        val tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = tadBuffers.getSecond();
        val devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        val tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);


        PointerPointer extraPointers = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                devTadShapeInfoZ, devTadOffsetsZ);

        val extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(op.z().dataType()), context) : null;


        val x = OpaqueNDArray.fromINDArray(op.x());
        val y = OpaqueNDArray.fromINDArray(op.y());
        val z = OpaqueNDArray.fromINDArray(op.z());
        val dim = Nd4j.createFromArray(dimension);
        val dimensionPointer = OpaqueNDArray.fromINDArray(dim);
        switch (op.getOpType()) {
            case SCALAR:
                Nd4j.getNativeOps().execScalarTad(extraPointers, op.opNum(),
                        x,
                        z,
                        y,
                        extraArgs,
                        dimensionPointer);

                break;
            case SCALAR_BOOL:
                Nd4j.getNativeOps().execScalarBoolTad(extraPointers, op.opNum(),
                        x,
                        z,
                        y,
                        extraArgs,
                        dimensionPointer);
                break;
            default:
                throw new UnsupportedOperationException();
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, null, st);

        return null;
    }

    @Override
    public INDArray exec(ScalarOp op) {
        invoke(op, null);
        return op.z();
    }

    protected CudaContext invoke(ScalarOp op, OpContext oc) {
        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);

        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);


        if(z == null){
            switch (op.getOpType()) {
                case SCALAR:
                    z = x.ulike();
                    setZ(x.ulike(), op, oc);
                    break;
                case SCALAR_BOOL:
                    z = Nd4j.createUninitialized(DataType.BOOL, x.shape());
                    setZ(z, op, oc);
                    break;
                default:
                    throw new ND4JIllegalStateException("Unknown op type: [" + op.getOpType() +"]");
            }
        }

        if (x.length() != z.length())
            throw new ND4JIllegalStateException("op.X length should be equal to op.Y length: ["
                    + Arrays.toString(x.shapeInfoDataBuffer().asInt()) + "] != ["
                    + Arrays.toString(z.shapeInfoDataBuffer().asInt()) + "]");

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        if (op.dimensions() != null) {
            intercept(op, op.dimensions().toLongVector());
            return null;
        }

        val context = AtomicAllocator.getInstance().getDeviceContext();

        val hostXShapeInfo = x == null ? null : AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer());
        val hostYShapeInfo = op.scalar() == null ? null : AddressRetriever.retrieveHostPointer(op.scalar().shapeInfoDataBuffer());
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(x.shapeInfoDataBuffer(), context);
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(op.getOpType() == Op.Type.SCALAR_BOOL ? x.dataType() : z.dataType()), context) : null;

        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(z.shapeInfoDataBuffer(), context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, null, null);

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(op.scalar());
        val zb = OpaqueNDArray.fromINDArray(z);

        switch (op.getOpType()) {
            case SCALAR_BOOL:
                Nd4j.getNativeOps().execScalarBool(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        zb,
                        yb,
                        extraArgs);
                break;
            case SCALAR:
                Nd4j.getNativeOps().execScalar(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        zb,
                        yb,
                        extraArgs);
                break;
            default:
                throw new UnsupportedOperationException("Unknown op type: " + op.getOpType());
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, oc, st);

        return null;
    }

    protected CudaContext invoke(TransformOp op, OpContext oc) {
        long st = profilingConfigurableHookIn(op);

        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        checkForCompression(op);


        AtomicAllocator allocator = AtomicAllocator.getInstance();

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val context = allocator.getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        // special temp array for IsMax along dimension
        INDArray ret = null;

        Pointer xShapeInfo = allocator.getPointer(x.shapeInfoDataBuffer(), context);


        Pointer dimensionDevPointer = null;
        Pointer dimensionHostPointer = null;
        Pointer retPointer = null;
        Pointer retHostShape = null;
        int dimension[] = null;

        val hostXShapeInfo = x == null ? null : AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer());
        var hostYShapeInfo = y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());


        if (z == null) {
            ret = Nd4j.createUninitialized(op.resultType(), x.shape(), x.ordering());
            setZ(ret, op, oc);
            z = ret;
        }

        var extraArgs = op.extraArgs() != null ? allocator.getPointer(op.extraArgsDataBuff(op.getOpType() == Op.Type.TRANSFORM_BOOL || op.getOpType() == Op.Type.PAIRWISE_BOOL ? x.dataType() : z.dataType()), context) : null;
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        Pointer hostTadShapeInfo = null;
        Pointer devTadShapeInfo = null;

        Pointer hostMaxTadShapeInfo = null;
        Pointer devMaxTadShapeInfo = null;

        Pair<DataBuffer, DataBuffer> tadBuffers;
        Pair<DataBuffer, DataBuffer> tadMaxBuffers;

        Pointer devTadOffsets = null;
        Pointer devMaxTadOffsets = null;

        op.validateDataTypes(oc, experimentalMode.get());

        Pointer zShapeInfo = allocator.getPointer(z.shapeInfoDataBuffer(), context);


        PointerPointer xShapeInfoHostPointer =
                extraz.get().put(AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), // 0
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
                        (Pointer) new CudaPointer(dimension == null ? 0 : dimension.length),
                        retHostShape);


        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);

        if (y != null) {
            Pointer yShapeInfo = allocator.getPointer(y.shapeInfoDataBuffer(), context);

            switch (op.getOpType()) {
                case TRANSFORM_BOOL:
                case PAIRWISE_BOOL:
                    Nd4j.getNativeOps().execPairwiseTransformBool(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            yb,
                            extraArgs,
                            zb );
                    break;
                default:
                    Nd4j.getNativeOps().execPairwiseTransform(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            yb,
                            zb,
                            extraArgs);
                    break;
            }
        } else {
            switch (op.getOpType()) {
                case TRANSFORM_ANY:
                    Nd4j.getNativeOps().execTransformAny(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_FLOAT:
                    Nd4j.getNativeOps().execTransformFloat(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_BOOL:
                    Nd4j.getNativeOps().execTransformBool(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_SAME:
                    Nd4j.getNativeOps().execTransformSame(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_STRICT:
                    Nd4j.getNativeOps().execTransformStrict(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        if (extraArgs != null)
            extraArgs.address();


        profilingConfigurableHookOut(op, oc, st);

        return null;
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
        return exec(op, null, rng);
    }

    public INDArray exec(RandomOp op, OpContext oc, Random rng) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        if(op instanceof BaseRandomOp && ((BaseRandomOp)op).isTripleArgRngOp() && z != null && x == null && y == null){
            //Ugly hack to ensure the triple arg call occurs
            //See GaussianDistribution.setZ etc
            x = z;
            y = z;
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);


        if (rng.getStatePointer() == null)
            throw new IllegalStateException(
                    "You should use one of NativeRandom classes for NativeOperations execution");

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val context = AtomicAllocator.getInstance().getDeviceContext();

        PointerPointer extraZZ = extraz.get().put(AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer()),
                context.getOldStream(), AtomicAllocator.getInstance().getDeviceIdPointer());


        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);

        if (x != null && y != null && z != null) {
            // triple arg call
            Nd4j.getNativeOps().execRandom3(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                    xb,
                    yb,
                    zb,
                    AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(z.dataType()), context));

        } else if (x != null && z != null) {
            //double arg call
            Nd4j.getNativeOps().execRandom2(extraZZ, op.opNum(),
                    rng.getStatePointer(), // rng state ptr
                    xb,
                    zb,
                    AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(z.dataType()),context));


        } else {
            // single arg call
            Nd4j.getNativeOps().execRandom(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                    zb,
                    AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(z.dataType()),context));
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, oc, st);

        return z;
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
            for (int i = 0; i < Nd4j.getNativeOps().getAvailableDevices(); i++) {
                Map<String, Object> deviceProps = new HashMap<>();

                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY, Nd4j.getNativeOps().getDeviceName(i));
                deviceProps.put(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY, Nd4j.getNativeOps().getDeviceFreeMemory(i));
                deviceProps.put(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY, Nd4j.getNativeOps().getDeviceTotalMemory(i));
                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_MAJOR_VERSION_KEY, (long) Nd4j.getNativeOps().getDeviceMajor(i));
                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_MINOR_VERSION_KEY, (long) Nd4j.getNativeOps().getDeviceMinor(i));

                devicesList.add(i, deviceProps);
            }

            // fill with basic general info
            props.put(Nd4jEnvironment.BACKEND_KEY, "CUDA");
            props.put(Nd4jEnvironment.CUDA_NUM_GPUS_KEY, Nd4j.getNativeOps().getAvailableDevices());
            props.put(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY, devicesList);
            props.put(Nd4jEnvironment.BLAS_VENDOR_KEY, (Nd4j.factory().blas()).getBlasVendor().toString());
            props.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

            // fill bandwidth information
            props.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());

            properties = props;
        } else {

            List<Map<String, Object>> devicesList = (List<Map<String, Object>>) properties.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);

            // just update information that might change over time
            for (int i = 0; i < Nd4j.getNativeOps().getAvailableDevices(); i++) {
                Map<String, Object> dev = devicesList.get(i);

                dev.put(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY, Nd4j.getNativeOps().getDeviceFreeMemory(i));
                dev.put(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY, Nd4j.getNativeOps().getDeviceTotalMemory(i));
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
    }

    @Override
    public void commit() {
        val ctx = AtomicAllocator.getInstance().getDeviceContext();
        ctx.syncOldStream();
        ctx.syncSpecialStream();
    }

    @Override
    public synchronized Map<String, CustomOpDescriptor> getCustomOperations() {
        if(customOps == null) {
            String list = Nd4j.getNativeOps().getAllCustomOps();

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



    protected LongShapeDescriptor getShapeFromPointer(LongPointer ptr) {
        val rank = (int) ptr.get(0);

        val shape = new long[rank * 2 + 4];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = ptr.get(i);
        }

        //val extras = ptr.get(Shape.shapeInfoLength(rank) - 3);
        val t = ArrayOptionsHelper.arrayType(shape);
        return LongShapeDescriptor.fromShape(Shape.shape(shape), Shape.stride(shape), Shape.elementWiseStride(shape), Shape.order(shape), ArrayOptionsHelper.dataType(shape), t == ArrayType.EMPTY);
    }

    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * PLEASE NOTE: right now this operations are executing on CPU
     * @param op
     */
    @Override
    public INDArray[] exec(CustomOp op) {

        Nd4j.getExecutioner().commit();

        boolean shapeOverride = false;
        if (op.numOutputArguments() == 0 && !op.isInplaceCall()) {
            try {
                val list = this.calculateOutputShape(op);
                if (list.isEmpty())
                    throw new ND4JIllegalStateException("Op name " + op.opName() + " failed to execute. You can't execute non-inplace CustomOp without outputs being specified");

                for (val shape: list)
                    op.addOutputArgument(Nd4j.create(shape, false));

                shapeOverride = true;
            } catch (Exception e) {
                throw new ND4JIllegalStateException("Op name " + op.opName() + " - no output arrays were provided and calculateOutputShape failed to execute", e);
            }
        }



        val name = op.opName();
        try (val context = (CudaOpContext) buildContext()) {
            // optionally skip shape validation on op execution
            if (shapeOverride)
                context.shapeFunctionOverride(true);

            context.markInplace(op.isInplaceCall());

            // transferring rng state
            context.setRngStates(Nd4j.getRandom().rootState(), Nd4j.getRandom().nodeState());

            //transferring input/output arrays
            context.setInputArrays(op.inputArguments());
            context.setOutputArrays(op.outputArguments());

            // transferring static args
            context.setBArguments(op.bArgs());
            context.setIArguments(op.iArgs());
            context.setTArguments(op.tArgs());
            context.setDArguments(op.dArgs());

            val result = exec(op, context);
            val states = context.getRngStates();


            // pulling states back
            Nd4j.getRandom().setStates(states.getFirst(), states.getSecond());

            return result;
        } catch (ND4JOpProfilerException e) {
            throw e;
        } catch (Exception e) {
            StringBuilder message = new StringBuilder();
            message.append("Op [" + name + "] execution failed with error " + "Cuda last error message: " + cudaGetErrorName(org.bytedeco.cuda.global.cublas.cublasGetError()).getString());
            throw new RuntimeException(message.toString(), e);
        }
    }


    @Override
    public ExecutionerType type() {
        return ExecutionerType.CUDA;
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        Preconditions.checkArgument(buffer instanceof CudaUtf8Buffer, "Expected Utf8Buffer");

        val addr = ((LongIndexer) buffer.indexer()).get(index);
        val ptr = new PagedPointer(addr);
        val str = new Nd4jCuda.utf8string(ptr);
        return str._buffer().capacity(str._length()).getString();
    }

    @Override
    public boolean isExperimentalMode() {
        return experimentalMode.get();
    }

    @Override
    public void scatterUpdate(ScatterUpdate.UpdateOp op, @NonNull INDArray array, @NonNull INDArray indices, @NonNull INDArray updates, long[] axis) {
        val context = AtomicAllocator.getInstance().getDeviceContext();

        val tadX = tadManager.getTADOnlyShapeInfo(array, axis);
        val tadY = tadManager.getTADOnlyShapeInfo(updates, axis);

        if (tadY.getSecond().length() != indices.length())
            throw new IllegalStateException("Number of updates doesn't match number of indices. Bad dimensions used?");

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val stuff = extraz.get().put(null, context.getOldStream());


        OpaqueNDArray xb = OpaqueNDArray.fromINDArray(array);
        OpaqueNDArray yb = OpaqueNDArray.fromINDArray(updates);
        OpaqueNDArray zb = OpaqueNDArray.fromINDArray(indices);
        INDArray axisArray = Nd4j.createFromArray(axis);
        OpaqueNDArray axis2 = OpaqueNDArray.fromINDArray(axisArray);
        Nd4j.getNativeOps().scatterUpdate(
                stuff,
                op.ordinal(),
                xb,zb,yb,axis2);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());
    }

    @Override
    public OpContext buildContext() {
        return new CudaOpContext();
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context) {
        Nd4j.getExecutioner().commit();
        long st = profilingConfigurableHookIn(op, context);
        if(op instanceof UserDefinedCustomOp) {
            ((UserDefinedCustomOp) op).exec(context);
            return context.getOutputArrays().toArray(new INDArray[0]);
        }



        val status = Nd4j.getNativeOps().execCustomOp2(null, op.opHash(), context.contextPointer());
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        if (status != 0)
            throw new RuntimeException("Op [" + op.opName() + "] execution failed");

        // check if input && output needs update
        for (val in:op.inputArguments()) {
            if (!in.isEmpty())
                ((BaseCudaDataBuffer) in.data()).actualizePointerAndIndexer();
        }

        for (val out:op.outputArguments()) {
            if (!out.isEmpty()) {
                ((BaseCudaDataBuffer) out.data()).actualizePointerAndIndexer();
                AtomicAllocator.getInstance().tickDeviceWrite(out);
            }

        }


        profilingConfigurableHookOut(op, context, st);

        if (context.getOutputArrays().isEmpty())
            return new INDArray[0];
        else
            return context.getOutputArrays().toArray(new INDArray[context.getOutputArrays().size()]);
    }

    @Override
    public INDArrayStatistics inspectArray(@NonNull INDArray array) {
        val debugInfo = new Nd4jCuda.DebugInfo();
        val ctx = AtomicAllocator.getInstance().getDeviceContext();
        AtomicAllocator.getInstance().synchronizeHostData(array);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val extras = extraz.get().put(
                null,
                ctx.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(),
                ctx.getBufferAllocation(),
                ctx.getBufferReduction(),
                ctx.getBufferScalar(),
                ctx.getBufferSpecial());


        Nd4j.getNativeOps().inspectArray(extras, AtomicAllocator.getInstance().getHostPointer(array), (LongPointer) AtomicAllocator.getInstance().getHostPointer(array.shapeInfoDataBuffer()), AtomicAllocator.getInstance().getPointer(array, ctx), (LongPointer) AtomicAllocator.getInstance().getPointer(array.shapeInfoDataBuffer()), debugInfo);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        return INDArrayStatistics.builder()
                .minValue(debugInfo._minValue())
                .maxValue(debugInfo._maxValue())
                .meanValue(debugInfo._meanValue())
                .stdDevValue(debugInfo._stdDevValue())
                .countInf(debugInfo._infCount())
                .countNaN(debugInfo._nanCount())
                .countNegative(debugInfo._negativeCount())
                .countPositive(debugInfo._positiveCount())
                .countZero(debugInfo._zeroCount())
                .build();
    }


    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val dbf = Nd4j.getNativeOps().shapeBuffer(shape.length, new LongPointer(shape), new LongPointer(stride), dtype.toInt(), order, elementWiseStride, empty);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val result = new CudaLongDataBuffer(Nd4j.getNativeOps().getConstantShapeBufferPrimary(dbf), Nd4j.getNativeOps().getConstantShapeBufferSpecial(dbf), Shape.shapeInfoLength(shape.length));


        return result;
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extras) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val dbf = Nd4j.getNativeOps().shapeBufferEx(shape.length, new LongPointer(shape), new LongPointer(stride), dtype.toInt(), order, elementWiseStride, extras);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val result = new CudaLongDataBuffer(Nd4j.getNativeOps().getConstantShapeBufferPrimary(dbf), Nd4j.getNativeOps().getConstantShapeBufferSpecial(dbf), Shape.shapeInfoLength(shape.length));


        return result;
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        OpaqueTadPack pack = Nd4j.getNativeOps().tadOnlyShapeInfo( array.shapeInfoDataBuffer().opaqueBuffer(), new LongPointer(ArrayUtil.toLongArray(dimension)), dimension.length);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val tadShape = new CudaLongDataBuffer(Nd4j.getNativeOps().getPrimaryShapeInfo(pack), Nd4j.getNativeOps().getSpecialShapeInfo(pack), Nd4j.getNativeOps().getShapeInfoLength(pack));
        val tadOffsets = new CudaLongDataBuffer(Nd4j.getNativeOps().getPrimaryOffsets(pack), Nd4j.getNativeOps().getSpecialOffsets(pack), Nd4j.getNativeOps().getNumberOfTads(pack));


        return new TadPack(tadShape, tadOffsets);
    }

    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val dbf = Nd4j.getNativeOps().constantBufferLong(desiredType.toInt(), new LongPointer(values), values.length);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val buffer = Nd4j.createBuffer(Nd4j.getNativeOps().getConstantDataBufferPrimary(dbf), Nd4j.getNativeOps().getConstantDataBufferSpecial(dbf), values.length, desiredType);
        buffer.setConstant(true);

        return buffer;
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType)  {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val dbf = Nd4j.getNativeOps().constantBufferDouble(desiredType.toInt(), new DoublePointer(values), values.length);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val buffer = Nd4j.createBuffer(Nd4j.getNativeOps().getConstantDataBufferPrimary(dbf), Nd4j.getNativeOps().getConstantDataBufferSpecial(dbf), values.length, desiredType);
        buffer.setConstant(true);

        return buffer;
    }




}


