/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.cpu.nativecpu.ops;


import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.nd4j.compression.impl.AbstractCompressor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.cpu.nativecpu.CpuTADManager;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.LongPointerWrapper;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jCpu;

import java.util.*;


/**
 *
 * Native operation
 * executioner in c++
 *
 * @author Adam Gibson
 */
@Slf4j
public class NativeOpExecutioner extends DefaultOpExecutioner {
    private NativeOps loop = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private ConstantHandler constantHandler = Nd4j.getConstantHandler();
    @Getter
    private CpuTADManager tadManager = new CpuTADManager();

    //thread locals for custom op inputs and outputs to prevent allocations
    //every time exec(CustomOp) is called
    private ThreadLocal<Map<Integer,PointerPointer>> inputShapes = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,PointerPointer>> inputBuffers = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,PointerPointer>> outputShapes = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,PointerPointer>> outputBuffers = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,DoublePointer>> tArgsPointer = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,ShortPointer>> halfArgsPointer = new ThreadLocal<>();



    protected Map<String, CustomOpDescriptor> customOps = null;

    protected ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();

    /**
     * Instead of allocating new memory chunks for each batch invocation, we reuse them on thread/opNum basis
     * Since for NativeOpExecutioner all executions are synchronous
     */
    private ThreadLocal<Map<Integer, Pointer>> batchPointers = new ThreadLocal<>();
    private ThreadLocal<Map<Integer, AggregateMemoryBlock>> memoryBlocks = new ThreadLocal<>();

    public NativeOpExecutioner() {
        tadManager.init(loop, constantHandler);
    }

    @Override
    public Op exec(Op op) {
        checkForCompression(op);

        if (op instanceof ScalarOp) {
            ScalarOp s = (ScalarOp) op;
            exec(s);
        }
        else if(op instanceof GradientOp) {
            op.exec();
        }
        else if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            exec(t);
        } else if (op instanceof ReduceOp) {
            ReduceOp ac = (ReduceOp) op;
            exec(ac);
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation iac = (IndexAccumulation) op;
            exec(iac); //Currently using DefaultOpExecutioner
        } else if (op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            exec(broadcastOp, broadcastOp.getDimension());
        }
        else if(op instanceof ShapeOp) {
            ShapeOp shapeOp = (ShapeOp) op;
            exec(shapeOp);
        } else if (op instanceof RandomOp) {
            RandomOp rngOp = (RandomOp) op;
            exec(rngOp, Nd4j.getRandom());
        }

        return op;
    }


    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        if (dimension == null || dimension.length == 0)
            dimension = new int[] {Integer.MAX_VALUE};

        checkForCompression(op);

        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        dimension = Shape.normalizeAxis(op.x().rank(), dimension);

        for (int i = 0; i < dimension.length; i++) {
            if (dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};

        boolean keepDims;
        boolean newFormat;
        if(op instanceof BaseIndexAccumulation) {
            keepDims = ((BaseIndexAccumulation) op).isKeepDims();
            newFormat = ((BaseIndexAccumulation) op).isNewFormat();
        } else {
            keepDims = false;
            newFormat = false;
        }
        long[] retShape = reductionShape(op.x(), dimension, newFormat, keepDims);


        if(op.z() == null || op.x() == op.z()) {
            val ret = Nd4j.create(retShape);

            op.setZ(ret);
        } else if(!Arrays.equals(retShape, op.z().shape())){
            throw new IllegalStateException("Z array shape does not match expected return type for op " + op
                    + ": expected shape " + Arrays.toString(retShape) + ", z.shape()=" + Arrays.toString(op.z().shape()));
        }


        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets);

        long st = profilingHookIn(op, tadBuffers.getFirst());

        Pointer x = op.x().data().addressPointer();
        Pointer z = op.z().data().addressPointer();

        if (op.z().isScalar()) {
            loop.execIndexReduceScalar(dummy, op.opNum(),
                        op.x().data().addressPointer(),
                        (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                        getPointerForExtraArgs(op), op.z().data().addressPointer(),
                        (LongPointer) op.z().shapeInfoDataBuffer().addressPointer());
            } else {
                loop.execIndexReduce(dummy, op.opNum(),
                        x,
                        (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                        getPointerForExtraArgs(op),
                        z,
                        (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                        (IntPointer) dimensionAddress,
                        dimension.length);
            }

        profilingHookOut(op, st);
        return op.z();
    }



    @Override
    public INDArray exec(ReduceOp op, int... dimension) {
        dimension = Shape.normalizeAxis(op.x().rank(), dimension);


        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        long[] maxShape = Shape.getMaxShape(op.x(),op.y());
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

        boolean keepDims;
        boolean newFormat;
        if(op instanceof BaseReduceOp) {
            keepDims = op.isKeepDims();
            newFormat = ((BaseReduceOp) op).isNewFormat();
        } else {
            keepDims = false;
            newFormat = false;
        }

        long[] retShape = reductionShape(op.x(), dimension, newFormat, keepDims);

        if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape) && ArrayUtil.prodLong(retShape) > 1 && op.y() == null)
            return op.noOp();

        /**
         * This is the result array.
         * We create it only if we hadn't provided it before
         */
        INDArray ret;
        if (op.z() == null || op.z() == op.x()) {
            if (op.isComplexAccumulation()) {
                long xT = op.x().tensorssAlongDimension(dimension);
                long yT = op.y().tensorssAlongDimension(dimension);

                ret = Nd4j.create(xT, yT);
            } else {
                if (op.y() != null) {

                    //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
                    if(op.x().lengthLong() == op.y().lengthLong()) {
                        //Pairwise
                        if (op.x().tensorssAlongDimension(dimension) != op.y().tensorssAlongDimension(dimension)) {
                            throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                                    Arrays.toString(op.x().shape()) + ", y shape = " + Arrays.toString(op.y().shape()) +
                                    ", dimension = " + Arrays.toString(dimension) + ")");
                        }
                    } else {
                        //Every X TAD vs. entirety of Y
                        val xTADSize = op.x().lengthLong() / op.x().tensorssAlongDimension(dimension);

                        if (xTADSize != op.y().length()) {
                            throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                                    " (x TAD size = " + xTADSize + ", y size = " + op.y().lengthLong());
                        }
                    }
                }

                ret = Nd4j.create(retShape);

            }
            op.setZ(ret);
        } else {
            // compare length
            long shapeProduct = (retShape.length == 0 ? 1 : ArrayUtil.prodLong(retShape));
            if (!op.isComplexAccumulation() && op.z().lengthLong() != shapeProduct)
                throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + Arrays.toString(retShape) + "]");
            else if (op.isComplexAccumulation()) {
                long xT = op.x().tensorssAlongDimension(dimension);
                long yT = op.y().tensorssAlongDimension(dimension);

                if (op.z().lengthLong() != xT * yT)
                    throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + (xT * yT) + "]");
            }

            if (op.x().data().dataType() == DataType.DOUBLE) {
                op.z().assign(op.zeroDouble());
            } else {
                op.z().assign(op.zeroFloat());
            }

            ret = op.z();
        }

        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);
        Pair<DataBuffer, DataBuffer> yTadBuffers = null;
        /**
         * Note that we use addresses in libnd4j.
         * We use reinterpret cast in c to take the long
         * we pass to JNI. This manages overhead.
         */
        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        // we're going to check, if that's TAD vs TAD comparison or TAD vs full array. if later - we're going slightly different route
        boolean tvf = false;
        if (op.y() != null) {
            if (op.x().tensorAlongDimension(0, dimension).lengthLong() == op.y().lengthLong()) {
                tvf = true;
            }
        }


        if (op.isComplexAccumulation()) {
            yTadBuffers = tadManager.getTADOnlyShapeInfo(op.y(), dimension);

            if (op.x().tensorAlongDimension(0, dimension).lengthLong() != op.y().tensorAlongDimension(0, dimension).lengthLong())
                throw new ND4JIllegalStateException("Impossible to issue AllDistances operation: TAD lengths mismatch along given dimension: " +
                        "x TAD length = " + op.x().tensorAlongDimension(0, dimension).lengthLong() + ", y TAD length " +
                        op.y().tensorAlongDimension(0, dimension).lengthLong());
        }


        /**
         * This is a pointer to a pointer in c.
         */
        //  FIXME: we need something better then 3rd element being non-null here...
        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, tvf ? hostTadOffsets : null);

        long st = profilingHookIn(op, tadBuffers.getFirst());

        /**
         * Note because dimension arrays don't change,
         * we use an {@link ConstantHandler} which knows how to reserve memory
         * for immutable buffers for the dimensions.
         * This gives us a pointer which is passed around in libnd4j.
         */
        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

            if (op instanceof Variance) {
                if (ret.isScalar()) {
                    loop.execSummaryStatsScalar(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            ((Variance) op).isBiasCorrected());
                } else {
                    Variance var = (Variance) op;
                    loop.execSummaryStats(dummy, op.opNum(), (DoublePointer) op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer) getPointerForExtraArgs(op),
                            (DoublePointer) op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer) dimensionAddress, dimension.length, var.isBiasCorrected());
                }

            }
            //pairwise reduction like similarity of two arrays
            else if (op.y() != null && op.getOpType() == Op.Type.REDUCE3) {
                if (op.isComplexAccumulation()) {
                    loop.execReduce3All(dummy, op.opNum(), (DoublePointer) op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer) getPointerForExtraArgs(op),
                            (DoublePointer) op.y().data().addressPointer(),
                            (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer) op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer) dimensionAddress, dimension.length,
                            (LongPointer) tadBuffers.getFirst().addressPointer(),
                            new LongPointerWrapper(tadBuffers.getSecond().addressPointer()),
                            (LongPointer) yTadBuffers.getFirst().addressPointer(),
                            new LongPointerWrapper(yTadBuffers.getSecond().addressPointer())
                    );
                } else if (ret.isScalar()) {
                            loop.execReduce3Scalar(dummy, op.opNum(),
                                    op.x().data().addressPointer(),
                                    (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                                    getPointerForExtraArgs(op),
                                    op.y().data().addressPointer(),
                                    (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                                    ret.data().addressPointer(),
                                    (LongPointer) ret.shapeInfoDataBuffer().addressPointer());
                } else {
                    loop.execReduce3(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer) dimensionAddress, dimension.length);
                }

            } else {
                if (ret.isScalar()) {
                            loop.execReduceFloat(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            ret.data().addressPointer(),
                            (LongPointer) ret.shapeInfoDataBuffer().addressPointer());
                } else {
                    loop.execReduceFloat(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer) dimensionAddress, dimension.length);
                }
            }

        return ret;
    }

    /**
     * ScalarOp along dimension
     * @param op
     * @param dimension
     */
    private void invoke(ScalarOp op, int[] dimension) {
        dimension = Shape.normalizeAxis(op.x().rank(), dimension);
        // do tad magic
        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;
        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         *
         * Note that this is the *result* TAD information. An op is always input (x) and output (z)
         * for result.
         * This is for assigning the result to of the operation along
         * the proper dimension.
         */
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, devTadShapeInfoZ, devTadOffsetsZ);


        loop.execScalar(dummy, op.opNum(),
                    op.x().data().addressPointer(),
                    (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                    op.z().data().addressPointer(),
                    (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                    op.y().data().addressPointer(),
                    (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                    getPointerForExtraArgs(op),
                    (IntPointer) Nd4j.getConstantHandler().getConstantBuffer(dimension).addressPointer(),
                    dimension.length);
    }

    private void exec(ScalarOp op) {
        if (executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        } else {
            long st = profilingHookIn(op);

            //validateDataType(Nd4j.dataType(), op);

            if (op.x().lengthLong() != op.z().lengthLong())
                throw new ND4JIllegalStateException("op.X length should be equal to op.Z length: " +
                        "x.length()=" + op.x().length() + ", z.length()=" + op.z().length() + " - x shape info = ["
                        + Arrays.toString(op.x().shapeInfoDataBuffer().asInt()) + "], z shape info = ["
                        + Arrays.toString(op.z().shapeInfoDataBuffer().asInt()) + "]");

            if (op.getDimension() != null) {
                invoke(op, op.getDimension());
                return;
            }


                    loop.execScalar(null,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                             op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            op.scalar().data().addressPointer(),
                            (LongPointer) op.scalar().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op));

            profilingHookOut(op, st);
        }
    }

    private Pointer getPointerForExtraArgs(Op op) {
        if (op.extraArgs() != null && op.extraArgsDataBuff() != null)
            return op.extraArgsDataBuff().addressPointer();
        return null;
    }

    private void exec(TransformOp op) {
        long st = 0;

//        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer dummy = extraz.get();

        // Pow operations might be special
        if (op.opNum() == 7) {
            if (op.y() != null && op.y().isScalar()) {
                op.setY(Nd4j.valueArrayOf(op.x().shape(), op.y().getDouble(0)));
            }
        }

        /**
         * This is the {@link org.nd4j.linalg.api.ops.impl.transforms.IsMax}
         * operation.
         *
         * @see {@link Op#extraArgs()}
         * for what an extra argument is in an op.
         *
         * The extra argument in the op here is the {@link org.nd4j.linalg.api.ops.impl.transforms.IsMax#IsMax(INDArray, int...)}
         * dimension to do the ismax along
         */
        if (op.opNum() == 41 && op.extraArgs() != null) {
            int[] dimension = new int[(int) op.extraArgs()[0]];

            for (int i = 0; i < dimension.length; i++) {
                dimension[i] = (int) op.extraArgs()[i + 1];
            }


            /**
             * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
             * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
             * The first item is the shape information. The second one is the offsets.
             */
            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.z(), dimension);


            Pointer tad = tadBuffers.getFirst().addressPointer();

            DataBuffer offsets = tadBuffers.getSecond();
            Pointer off = offsets == null ? null : offsets.addressPointer();
            dummy.put(0, tad);
            dummy.put(1, off);

            st = profilingHookIn(op, tadBuffers.getFirst());
        } else
            st = profilingHookIn(op);

            if (op.y() != null) {

                int xEWS = op.x().elementWiseStride();
                int yEWS = op.y().elementWiseStride();
                int zEWS = op.z().elementWiseStride();

                boolean xRow = op.x().isRowVector();
                boolean yRow = op.y().isRowVector();
                boolean zRow = op.z().isRowVector();

                if (op.x().length() != op.y().length() || op.x().length() != op.z().length())
                    throw new ND4JIllegalStateException("X, Y and Z arguments should have the same length for PairwiseTransform " +
                            op.opName() + ". x: length " + op.x().length() + ", shape " + Arrays.toString(op.x().shape()) +
                            "; y: " + op.y().length() + ", shape " + Arrays.toString(op.y().shape()) +
                            "; z: " + op.z().length() + ", shape " + Arrays.toString(op.z().shape()));

                    loop.execPairwiseTransform(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            op.y().data().addressPointer(),
                            (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op));
            } else {
                if (op instanceof FloatTransformOp) {
                    loop.execTransformFloat(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op));
                } else if (op instanceof StrictTransformOp) {
                    loop.execTransformStrict(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op));
                } else if (op instanceof SameTransformOp) {
                    loop.execTransformSame(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op));
                } else if (op instanceof BoolTransformOp) {
                    loop.execTransformBool(dummy, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op));
                }


            }

        profilingHookOut(op, st);
    }

    @Override
    public INDArray exec(BroadcastOp op, int... dimension) {
        long st = profilingHookIn(op);
        if(dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};
        dimension = Shape.normalizeAxis(op.x().rank(), dimension);

        validateDataType(Nd4j.dataType(), op);

        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank() && dimension[i] != Integer.MAX_VALUE)
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");
        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        //        if (!Arrays.equals(op.x().shape(),op.z().shape()) || !Arrays.equals(op.x().stride(),op.z().stride()) || op.x().ordering() != op.z().ordering()) {
        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();
        /*
        log.info("Broascast dimension: {}", Arrays.toString(dimension));
        log.info("x shape: {}; x TAD: {}; comp TAD: {}", Arrays.toString(op.x().shapeInfoDataBuffer().asInt()), Arrays.toString(tadBuffers.getFirst().asInt()), Arrays.toString(op.x().tensorAlongDimension(0, dimension).shapeInfoDataBuffer().asInt()));
        log.info("z shape: {}; z TAD: {}", Arrays.toString(op.z().shapeInfoDataBuffer().asInt()), Arrays.toString(tadBuffersZ.getFirst().asInt()));
        log.info("y shape: {}", Arrays.toString(op.y().shapeInfoDataBuffer().asInt()));
        log.info("-------------");
        */

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, devTadShapeInfoZ, devTadOffsetsZ);

        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

            loop.execBroadcast(dummy, op.opNum(),
                    op.x().data().addressPointer(),
                    (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                    op.y().data().addressPointer(),
                    (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                    op.z().data().addressPointer(),
                    (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                    (IntPointer) dimensionAddress,
                    dimension.length);


        return op.z();
    }

    private void exec(IndexAccumulation op) {
        if (executionMode() == ExecutionMode.JAVA) {
            super.exec(op);

        } else {
            if(op.z() == op.x() || op.z() == null) {
                op.setZ(Nd4j.scalar(0.0));
            }

            long st = profilingHookIn(op);

            validateDataType(Nd4j.dataType(), op);

            loop.execIndexReduceScalar(null, op.opNum(),
                        op.x().data().addressPointer(),
                        (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                        getPointerForExtraArgs(op),
                        op.z().data().addressPointer(),
                        (LongPointer) op.z().shapeInfoDataBuffer().addressPointer());

            profilingHookOut(op, st);
        }
    }

    private void exec(ReduceOp op) {
        if (executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        }
        else if(op.isExecSpecial()) {
            op.exec();
        }
        else {
            long st = profilingHookIn(op);

            validateDataType(Nd4j.dataType(), op);

            if(op.z() == op.x()) {
                op.setZ(Nd4j.scalar(0.0));
            }

            // since we're going to call reduceToScalar, we must ensure equal lengths
            if (op.y() != null && op.getOpType() == Op.Type.REDUCE3) {
                if (op.x().lengthLong() != op.y().lengthLong())
                    throw new ND4JIllegalStateException("X and Y operands should have equal lengths. X length: " + op.x().lengthLong() +
                            ", X shape: " + Arrays.toString(op.x().shape()) + "; Y length: " + op.y().lengthLong() +
                            ", Y shape: " + Arrays.toString(op.y().shape()));
            }

                if (op instanceof Variance) {
                    loop.execSummaryStatsScalar(null, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                           true);
                } else if (op.y() != null && op.getOpType() == Op.Type.REDUCE3) {
                    loop.execReduce3Scalar(null, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(), op.z().data().addressPointer(), (LongPointer) op.z().shapeInfoDataBuffer().addressPointer());
                } else {
                    loop.execReduceFloat(null, op.opNum(),
                            op.x().data().addressPointer(),
                            (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op), op.z().data().addressPointer(), (LongPointer) op.z().shapeInfoDataBuffer().addressPointer());
                }


            profilingHookOut(op, st);
        }
    }


    protected <T extends Aggregate> Pointer getPointer(Batch<T> batch) {
        if (batchPointers.get() == null)
            batchPointers.set(new HashMap<Integer, Pointer>());

        if (!batchPointers.get().containsKey(batch.opNum())) {
            val pointer = new IntPointer(batch.getSample().getRequiredBatchMemorySize() / 4 );
            batchPointers.get().put(batch.opNum(), pointer);
            return pointer;
        }

        return batchPointers.get().get(batch.opNum());
    }


    /**
     * This method executes previously built batch
     *
     * @param batch
     */
    @Override
    public <T extends Aggregate> void exec(Batch<T> batch) {
        //profilingHookIn(batch);

        IntPointer pointer = (IntPointer) getPointer(batch);

        int maxTypes = 5;

        int maxIntArrays = batch.getSample().maxIntArrays();

        int maxArraySize = batch.getSample().maxIntArraySize();


        int indexPos = maxTypes * Batch.getBatchLimit();
        int intArraysPos = indexPos + (batch.getSample().maxIndexArguments() * Batch.getBatchLimit());
        int realPos = (intArraysPos + (maxIntArrays * maxArraySize * Batch.getBatchLimit()))
                / (Nd4j.dataType() == DataType.DOUBLE ? 2 : 1);
        int argsPos = (realPos + ((batch.getSample().maxRealArguments() * Batch.getBatchLimit())))
                / (Nd4j.dataType() == DataType.DOUBLE ? 1 : 2);
        int shapesPos = argsPos + (batch.getSample().maxArguments() * Batch.getBatchLimit());
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

            if (Nd4j.dataType() == DataType.FLOAT) {
                FloatPointer fPtr = new FloatPointer(pointer);
                for (int e = 0; e < op.getRealArguments().size(); e++) {
                    idx = realPos + i * op.maxRealArguments();
                    fPtr.put(idx + e, op.getRealArguments().get(e).floatValue());
                }
            } else if (Nd4j.dataType() == DataType.DOUBLE) {
                DoublePointer dPtr = new DoublePointer(pointer);
                for (int e = 0; e < op.getRealArguments().size(); e++) {
                    idx = realPos + (i * op.maxRealArguments());
                    dPtr.put(idx + e, op.getRealArguments().get(e).doubleValue());
                }
            }

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            // putting arguments pointers

            PointerPointer ptrPtr = new PointerPointer(pointer);//extraz.get().put(pointer);

            for (int e = 0; e < op.getArguments().size(); e++) {
                idx = argsPos + i * batch.getSample().maxArguments();

                if (op.getArguments().get(e) != null) {
                    ptrPtr.put(idx + e, op.getArguments().get(e).data().addressPointer());
                }
            }


            // putting shape pointers
            for (int e = 0; e < op.getShapes().size(); e++) {
                idx = shapesPos + i * batch.getSample().maxShapes();

                if (op.getShapes().get(e) != null)
                    ptrPtr.put(idx + e, op.getShapes().get(e).addressPointer());
            }
        }

        loop.execAggregateBatch(null, batch.getNumAggregates(), batch.opNum(),
                    batch.getSample().maxArguments(), batch.getSample().maxShapes(),
                    batch.getSample().maxIntArrays(), batch.getSample().maxIntArraySize(),
                    batch.getSample().maxIndexArguments(), batch.getSample().maxRealArguments(), pointer, 0);

    }

    /**
     * This method takes arbitrary
     * sized list of {@link Aggregate},
     * and packs them into batches
     * Note here that this is mainly used for random number generation
     * for {@link RandomOp} and things like {@link org.nd4j.linalg.api.rng.distribution.Distribution}
     * @param batch the list of {@link Aggregate} to
     *              execute upon
     */
    @Override
    public void exec(List<Aggregate> batch) {
        if (batch.size() == 0)
            return;

        List<Batch<Aggregate>> batches = Batch.getBatches(batch);
        for (Batch<Aggregate> single : batches) {
            this.exec(single);
        }
    }

    /**
     * This method takes arbitrary
     * sized list of {@link Aggregate},
     * and packs them into batches
     * Note here that this is mainly used for random number generation
     * for {@link RandomOp} and things like {@link org.nd4j.linalg.api.rng.distribution.Distribution}
     * @param op the list of {@link Aggregate} to
     *              execute upon
     */
    @Override
    public void exec(Aggregate op) {
        // long st = profilingHookIn(op);

        if (memoryBlocks.get() == null)
            memoryBlocks.set(new HashMap<Integer, AggregateMemoryBlock>());

        if (memoryBlocks.get().get(op.opNum()) == null)
            memoryBlocks.get().put(op.opNum(), new AggregateMemoryBlock(op));

        AggregateMemoryBlock block = memoryBlocks.get().get(op.opNum());

        int numArguments = op.getArguments().size();
        int numIndexArguments = op.getIndexingArguments().size();
        int numRealArguments = op.getRealArguments().size();
        int numShapes = op.getShapes().size();
        int numIntArrays = op.getIntArrayArguments().size();

        PointerPointer arguments = block.getArgumentsPointer(); //new PointerPointer(numArguments);
        List<IntPointer> pointers = new ArrayList<>();
        PointerPointer intArrays = block.getArraysPointer(); //new PointerPointer(numIntArrays);

        for (int x = 0; x < numArguments; x++) {
            arguments.put(x, op.getArguments().get(x) == null ? null
                    : op.getArguments().get(x).data().addressPointer());
        }

        PointerPointer shapes = block.getShapesPointer(); //new PointerPointer(numShapes);

        for (int x = 0; x < numShapes; x++) {
            if (op.getShapes().get(x).dataType() != DataType.INT)
                throw new RuntimeException("ShapeBuffers should have INT data opType");

            shapes.put(x, op.getShapes().get(x) == null ? null : op.getShapes().get(x).addressPointer());
        }

        //int[] indexes = new int[numIndexArguments];
        IntPointer pointer = block.getIndexingPointer();
        for (int x = 0; x < numIndexArguments; x++) {
            pointer.put(x, op.getIndexingArguments().get(x));
        }

        //IntPointer pointer = new IntPointer(indexes);

        double[] reals = new double[numRealArguments];
        for (int x = 0; x < numRealArguments; x++) {
            //reals[x] = op.getRealArguments().get(x).doubleValue();
            if (Nd4j.dataType() == DataType.FLOAT)
                ((FloatPointer) block.getRealArgumentsPointer()).put(x, op.getRealArguments().get(x).floatValue());
            else
                ((DoublePointer) block.getRealArgumentsPointer()).put(x, op.getRealArguments().get(x).doubleValue());
        }

        for (int x = 0; x < numIntArrays; x++) {
            IntPointer intPtr = block.getIntArrays().get(x); //new IntPointer(op.getIntArrayArguments().get(x));
            intPtr.put(op.getIntArrayArguments().get(x), 0, op.getIntArrayArguments().get(x).length);
            intArrays.put(x, intPtr);
            pointers.add(intPtr);
        }

        //INDArray realsBuffer = Nd4j.create(reals);



        loop.execAggregate(null, op.opNum(), arguments, numArguments, shapes, numShapes, pointer,
                    numIndexArguments, intArrays, numIntArrays, (DoublePointer) block.getRealArgumentsPointer(),
                    numRealArguments, 0);

    }

    /**
     * This method return set of key/value and
     * key/key/value objects,
     * describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        Properties properties = super.getEnvironmentInformation();
        properties.put(Nd4jEnvironment.BACKEND_KEY, "CPU");
        properties.put(Nd4jEnvironment.OMP_THREADS_KEY, loop.ompGetMaxThreads());
        properties.put(Nd4jEnvironment.BLAS_THREADS_KEY, Nd4j.factory().blas().getMaxThreads());
        properties.put(Nd4jEnvironment.BLAS_VENDOR_KEY, (Nd4j.factory().blas()).getBlasVendor().toString());
        properties.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

        // fill bandwidth information
        /*
        Note: Environment information is logged as part of ND4J initialization... but PerformanceTracker required
        ND4J init to be completed before it can be initialized. Hence we can get a null PerformanceTracker when
        OpExecutioner.printEnvironmentInformation() is called as part of ND4J class initialization - even
        though PerformanceTracker.getInstance() refers to a static final field (as it may not yet be initialized)
         */
        if(PerformanceTracker.getInstance() != null) {
            properties.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());
        }

        return properties;
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

    /**
     * This method executes specific
     * RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    @Override
    public INDArray exec(RandomOp op, Random rng) {
        if (rng.getStateBuffer() == null)
            throw new IllegalStateException(
                    "You should use one of NativeRandom classes for NativeOperations execution. Op class: " + op.getClass().getName());

        long st = profilingHookIn(op);

        validateDataType(Nd4j.dataType(), op);

        if (op.x() != null && op.y() != null && op.z() != null) {
            // triple arg call
            loop.execRandom(null, op.opNum(), rng.getStatePointer(), // rng state ptr
                        op.x().data().addressPointer(),
                        (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                        op.y().data().addressPointer(),
                        (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(),
                        op.z().data().addressPointer(),
                        (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                        op.extraArgsDataBuff().addressPointer());
        } else if (op.x() != null && op.z() != null) {
            //double arg call
                loop.execRandom(null, op.opNum(), rng.getStatePointer(), // rng state ptr
                        op.x().data().addressPointer(),
                        (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                        op.z().data().addressPointer(),
                        (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                        (DoublePointer) op.extraArgsDataBuff().addressPointer());
        } else {
            // single arg call
                loop.execRandom(null, op.opNum(), rng.getStatePointer(), // rng state ptr
                        op.z().data().addressPointer(),
                        (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                        op.extraArgsDataBuff().addressPointer());
        }

        profilingHookOut(op, st);

        return op.z();
    }

    @Override
    public TADManager getTADManager() {
        return tadManager;
    }

    /**
     * This class holds memory chunks required for single specific Aggregate op.
     * Can be used together with ThreadLocal variables
     */
    @Data
    private static class AggregateMemoryBlock {
        private List<IntPointer> intArrays = new ArrayList<>();
        private IntPointer indexingPointer;
        private Pointer realArgumentsPointer;
        private PointerPointer shapesPointer;
        private PointerPointer argumentsPointer;
        private PointerPointer arraysPointer;

        private final int opNum;

        private AggregateMemoryBlock(@NonNull Aggregate op) {

            opNum = op.opNum();

            // creating IntArrays
            for (int i = 0; i < op.maxIntArrays(); i++) {
                intArrays.add(new IntPointer(op.maxIntArraySize()));
            }

            // allocating chunk for IndexingArguments
            indexingPointer = new IntPointer(op.maxIndexArguments());

            // allocating chunk for RealArguments
            realArgumentsPointer = Nd4j.dataType() == DataType.DOUBLE ? new DoublePointer(op.maxRealArguments())
                    : new FloatPointer(op.maxRealArguments());

            // allocating chunk for shapesPointer
            shapesPointer = new PointerPointer(op.maxShapes());

            // allocating chunk for argumentsPointer
            argumentsPointer = new PointerPointer(op.maxArguments());

            // chunk for intArrays
            arraysPointer = new PointerPointer(op.maxIntArrays());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;

            AggregateMemoryBlock that = (AggregateMemoryBlock) o;

            return opNum == that.opNum;
        }

        @Override
        public int hashCode() {
            return opNum;
        }
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return thresholdEncode(input, threshold, null);
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {

        //val condition = new MatchCondition(input, Conditions.absGreaterThanOrEqual(threshold));
        //long t1 = System.currentTimeMillis();
        int cntAbs = loop.estimateThreshold(null,
                input.data().addressPointer(),
                (LongPointer) input.shapeInfoDataBuffer().addressPointer(),
                (int) input.length(),
                (float) threshold);
        //long t2 = System.currentTimeMillis();

        if (cntAbs < 2)
            return null;

        if (boundary != null)
            cntAbs = Math.min(cntAbs, boundary);

        //log.info("S: {}; T: {}", cntAbs, t2 - t1);

        DataBuffer buffer = input.data();

        long originalLength = buffer.length() * Nd4j.sizeOfDataType(buffer.dataType());
        int compressedLength = cntAbs + 4;
        // first 3 elements contain header

        DataBuffer encodedBuffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createInt(4+cntAbs, false) : Nd4j.getDataBufferFactory().createInt(4+cntAbs, false, Nd4j.getMemoryManager().getCurrentWorkspace());

        encodedBuffer.put(0, cntAbs);
        encodedBuffer.put(1, (int) buffer.length());
        encodedBuffer.put(2, Float.floatToIntBits((float) threshold));

        // format id
        encodedBuffer.put(3, ThresholdCompression.FLEXIBLE_ENCODING);

        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(compressedLength * 4); // sizeOf(INT)
        descriptor.setOriginalLength(originalLength);
        descriptor.setOriginalElementSize(Nd4j.sizeOfDataType(buffer.dataType()));
        descriptor.setNumberOfElements(buffer.length());

        descriptor.setCompressionAlgorithm("THRESHOLD");
        descriptor.setCompressionType(CompressionType.LOSSLESS);

        //CompressedDataBuffer cbuff = new CompressedDataBuffer(pointer, descriptor);

        Nd4j.getNDArrayFactory().convertDataEx(AbstractCompressor.getBufferTypeEx(buffer), buffer.addressPointer(), DataTypeEx.THRESHOLD, encodedBuffer.addressPointer(), buffer.length());

        Nd4j.getAffinityManager().tagLocation(buffer, AffinityManager.Location.HOST);

        return Nd4j.createArrayFromShapeBuffer(encodedBuffer, input.shapeInfoDataBuffer());
    }

    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        DataBuffer buffer = encoded.data();

        if (buffer.dataType() != DataType.INT)
            throw new ND4JIllegalStateException("thresholdEncoded array should have dataType of INT");

        long compressedLength = buffer.getInt(0);
        long originalLength = buffer.getInt(1);
        float threshold = buffer.getInt(2);

        if (target.lengthLong() != originalLength)
            throw new ND4JIllegalStateException("originalLength ["+ originalLength+"] stored in encoded array doesn't match target length ["+ target.lengthLong()+"]");

        DataTypeEx typeDst = AbstractCompressor.getBufferTypeEx(target.data());

        loop.convertTypes(null, DataTypeEx.THRESHOLD.ordinal(), buffer.addressPointer(), target.length(), typeDst.ordinal(), target.data().addressPointer());

        return target;
    }


    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        long length = indArray.lengthLong();
        long tLen = target.data().length();

        if (tLen != (length / 16 + 5))
            throw new ND4JIllegalStateException("Length of target array should be " + (length / 16 + 5));

        if (target.data().dataType() != DataType.INT)
            throw new ND4JIllegalStateException("Target array should have INT dataType");

        DataBuffer buffer = target.data();

        buffer.put(0, (int) length);
        buffer.put(1, (int) length);
        buffer.put(2, Float.floatToIntBits((float) threshold));

        // format id
        buffer.put(3, ThresholdCompression.BITMAP_ENCODING);

        long affected = loop.encodeBitmap(null,
                indArray.data().addressPointer(),
                (LongPointer) indArray.shapeInfoDataBuffer().addressPointer(),
                length,
                (IntPointer) buffer.addressPointer(),
                (float) threshold);

        return affected;
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {

        loop.decodeBitmap(null,
                encoded.data().addressPointer(),
                target.length(),
                target.data().addressPointer(),
                (LongPointer) target.shapeInfoDataBuffer().addressPointer()
        );

        return target;
    }


    @Override
    public synchronized Map<String, CustomOpDescriptor> getCustomOperations() {
        if (customOps == null) {
            String list = loop.getAllCustomOps();

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


    private PointerPointer getPointerPointerFrom(ThreadLocal<Map<Integer,PointerPointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,PointerPointer> store = new HashMap<>();
            store.put(numArguments,new PointerPointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            PointerPointer pointerPointer = new PointerPointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }




    private ShortPointer getShortPointerFrom(ThreadLocal<Map<Integer,ShortPointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,ShortPointer> store = new HashMap<>();
            store.put(numArguments,new ShortPointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            ShortPointer pointerPointer = new ShortPointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }



    private DoublePointer getDoublePointerFrom(ThreadLocal<Map<Integer,DoublePointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,DoublePointer> store = new HashMap<>();
            store.put(numArguments,new DoublePointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            DoublePointer pointerPointer = new DoublePointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }


    private PointerPointer getInputShapes(int numArguments) {
       return getPointerPointerFrom(inputShapes,numArguments);
    }

    private PointerPointer getInputBuffers(int numArguments) {
        return getPointerPointerFrom(inputBuffers,numArguments);

    }

    private PointerPointer getOutputShapes(int numArguments) {
        return getPointerPointerFrom(outputShapes,numArguments);

    }

    private PointerPointer getOutputBuffers(int numArguments) {
        return getPointerPointerFrom(outputBuffers,numArguments);

    }

    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * @param op
     */
    public void exec(@NonNull CustomOp op) {
        long st = profilingHookIn(op);

        if (op.numOutputArguments() == 0 && !op.isInplaceCall()) {
            try {
                val list = this.calculateOutputShape(op);
                if (list.isEmpty())
                    throw new ND4JIllegalStateException("Op name " + op.opName() + " failed to execute. You can't execute non-inplace CustomOp without outputs being specified");

                for (val shape: list)
                    op.addOutputArgument(Nd4j.create(shape));

            } catch (Exception e) {
                throw new ND4JIllegalStateException("Op name " + op.opName() + " failed to execute. You can't execute non-inplace CustomOp without outputs being specified");
            }
        }

        val name = op.opName().toLowerCase();
        val hash = op.opHash();


        val inputShapes = getInputShapes(op.numInputArguments());
        val inputBuffers = getInputBuffers(op.numInputArguments());

        int cnt= 0;
        val inputArgs = op.inputArguments();
        for (val in: inputArgs) {
            if(in == null)
                throw new NullPointerException("Input argument is null for op " + op.getClass().getName());

            if (!in.isEmpty())
                inputBuffers.put(cnt, in.data().addressPointer());

            inputShapes.put(cnt++, in.shapeInfoDataBuffer().addressPointer());
        }

        val outputArgs = op.outputArguments();
        for(int i = 0; i < outputArgs.length; i++) {
            if(outputArgs[i] == null)
                throw new ND4JIllegalStateException("Op output arguments must not be null! Op " + op.getClass().getName());
        }


        val outputShapes = getOutputShapes(op.numOutputArguments());
        val outputBuffers = getOutputBuffers(op.numOutputArguments());

        cnt= 0;
        for (val out: outputArgs) {
            if(out.isEmpty()){
                outputBuffers.put(cnt, null);
            } else {
                outputBuffers.put(cnt, out.data().addressPointer());
            }
            outputShapes.put(cnt++, out.shapeInfoDataBuffer().addressPointer());
        }



        val iArgs = op.numIArguments() > 0 ? new LongPointer(op.numIArguments()) : null;
        cnt = 0;
        val iArgs1 = op.iArgs();
        for (val i: iArgs1)
            iArgs.put(cnt++, i);


            val tArgs = op.numTArguments() > 0 ? getDoublePointerFrom(tArgsPointer,op.numTArguments()) : null;
            val tArgs1 = op.tArgs();

            cnt = 0;
            for (val t: tArgs1)
                tArgs.put(cnt++, t);

            val t = op.numInputArguments();

            OpStatus status = OpStatus.ND4J_STATUS_OK;
            try {
                status = OpStatus.byNumber(loop.execCustomOp(
                        null,
                        hash,
                        inputBuffers,
                        inputShapes,
                        op.numInputArguments(),
                        outputBuffers,
                        outputShapes,
                        op.numOutputArguments(),
                        tArgs, op.numTArguments(),
                        iArgs, op.numIArguments(),
                        op.isInplaceCall()));
            }catch(Exception e) {
                log.error("Failed to execute op " + op.opName() + ". Attempted to execute with " +
                                String.valueOf(op.numInputArguments()) + " inputs, " +
                                String.valueOf(op.numOutputArguments()) + " outputs, "+
                                String.valueOf(op.numTArguments()) + " targs and " +
                                String.valueOf(op.numIArguments()) + " iargs. " +
                "Please see above message (printed out from c++) for a possible cause of error.");
                throw e;
            }


        profilingHookOut(op, st);
    }

    protected long[] getShapeFromPointer(LongPointer ptr) {
        val rank = (int) ptr.get(0);
        long[] array = new long[rank];
        for (int i = 0; i < rank; i++) {
            array[i] = ptr.get(i+1);
        }
        return array;
    }

    @Override
    public List<long[]> calculateOutputShape(@NonNull CustomOp op) {
        val lc = op.opName().toLowerCase();
        val hash = op.opHash();

        val result = new ArrayList<long[]>();
        if(op.numInputArguments() < 1 && op.getDescriptor().getNumInputs() != -2) {
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: number of input args was 0",
                        op.getClass().getName());
            }
            return Collections.emptyList();
        }


        val inputBuffers = new PointerPointer<>(op.numInputArguments());
        val inputShapes = new PointerPointer<>(op.numInputArguments());
        val inputArgs = op.inputArguments();
        int cnt= 0;
        for (val in: inputArgs) {
            if (!in.isEmpty())
                inputBuffers.put(cnt, in.data().addressPointer());

            inputShapes.put(cnt++, in.shapeInfoDataBuffer().addressPointer());
        }


        val iArgs = op.numIArguments() > 0 ? new LongPointer(op.numIArguments()) : null;
        cnt = 0;
        val iArgs1 = op.iArgs();
        for (val i: iArgs1)
            iArgs.put(cnt++, i);

            val tArgs = op.numTArguments() > 0 ? new DoublePointer(op.numTArguments()) : null;

            cnt = 0;
            val tArgs1 = op.tArgs();
            for (val t: tArgs1)
                tArgs.put(cnt++, t);

            val ptrptr= (Nd4jCpu.ShapeList) loop.calculateOutputShapes(null,
                    hash, inputBuffers, inputShapes, op.numInputArguments(), tArgs,
                    op.numTArguments(), iArgs, op.numIArguments());

            if (ptrptr == null)
                throw new RuntimeException();

            for (int e = 0; e < ptrptr.size(); e++ )
                result.add(getShapeFromPointer(new PagedPointer(ptrptr.at(e)).asLongPointer()));


            loop.deleteShapeList(ptrptr);

        if(log.isTraceEnabled()){
            String[] arr = new String[result.size()];
            for( int i=0; i<result.size(); i++ ){
                arr[i] = Arrays.toString(result.get(i));
            }
            log.trace("Calculated output shapes for op {} - {}", op.getClass().getName(), Arrays.toString(arr));
        }
        return result;
    }


    @Override
    public void enableDebugMode(boolean reallyEnable) {
        debug.set(reallyEnable);
        loop.enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        verbose.set(reallyEnable);
        loop.enableVerboseMode(reallyEnable);
    }


    @Override
    public void registerGraph(long id, Pointer graph) {
         loop.registerGraph(null, id, graph);
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, @NonNull Map<String, INDArray> map, @NonNull Map<String, Integer> reverseMap) {

        val ptrBuffers = new PointerPointer(map.size());
        val ptrShapes = new PointerPointer(map.size());
        val ptrIndices = new IntPointer(map.size());

        int cnt = 0;
        val keySet = new ArrayList<String>(map.keySet());
        for (val key: keySet) {
            val array = map.get(key);

            ptrBuffers.put(cnt, array.data().addressPointer());
            ptrShapes.put(cnt, array.shapeInfoDataBuffer().addressPointer());
            ptrIndices.put(cnt, reverseMap.get(key));

            cnt++;
        }

        val newMap = new LinkedHashMap<String, INDArray>();

            val result = (Nd4jCpu.VariablesSet) loop.executeStoredGraph(null, id, ptrBuffers, ptrShapes, ptrIndices, map.size());

            val status = OpStatus.byNumber(result.status());

            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);

            for (int e = 0; e < result.size(); e++) {
                val var = result.at(e);
                val nodeId = var.id();
                val index = var.index();
                val shapeInfo = var.getNDArray().shapeInfo();
                val buffer = var.getNDArray().buffer();

                val rank = (int) shapeInfo.get(0);
                val jshape = new long[rank * 2 + 4];
                for (int i = 0; i < jshape.length; i++) {
                    jshape[i] = shapeInfo.get(i);
                }

                val shapeOf = Shape.shapeOf(jshape);
                val stridesOf = Shape.stridesOf(jshape);
                val order = Shape.order(jshape);
                val array = Nd4j.create(shapeOf, stridesOf, 0, order);

                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(array.data().addressPointer(), buffer, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

                //newMap.put(keySet.get(nodeId), array);
                val nodeName = var.getName().getString();
                newMap.put(nodeName, array);
            }

        loop.deleteVariablesSet(result);

        return newMap;
    }

    @Override
    public void forgetGraph(long id) {
        loop.unregisterGraph(null, id);
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
        loop.setElementThreshold(threshold);
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
        loop.setTADThreshold(threshold);
    }


    private static long[] reductionShape(INDArray x, int[] dimension, boolean newFormat, boolean keepDims){
        boolean wholeArray = Shape.wholeArrayDimension(dimension);
        long[] retShape;
        if(!newFormat) {
            retShape = wholeArray ? new long[] {1, 1} : ArrayUtil.removeIndex(x.shape(), dimension);

            //ensure vector is proper shape (if old format)
            if (retShape.length == 1) {
                if (dimension[0] == 0)
                    retShape = new long[]{1, retShape[0]};
                else
                    retShape = new long[]{retShape[0], 1};
            } else if (retShape.length == 0) {
                retShape = new long[]{1, 1};
            }
        } else {
            if(keepDims){
                retShape = x.shape().clone();
                if(wholeArray){
                    for( int i=0; i<retShape.length; i++ ){
                        retShape[i] = 1;
                    }
                } else {
                    for (int d : dimension) {
                        retShape[d] = 1;
                    }
                }
            } else {
                retShape = wholeArray ? new long[0] : ArrayUtil.removeIndex(x.shape(), dimension);
            }
        }
        return retShape;
    }

    @Override
    public ExecutionerType type() {
        return ExecutionerType.NATIVE_CPU;
    }
}
