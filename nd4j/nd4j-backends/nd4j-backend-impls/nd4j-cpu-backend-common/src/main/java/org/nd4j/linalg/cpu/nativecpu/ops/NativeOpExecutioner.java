/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.cpu.nativecpu.ops;


import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.config.ND4JEnvironmentVars;
import org.nd4j.linalg.api.buffer.*;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.any.Assign;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.cpu.nativecpu.CpuTADManager;
import org.nd4j.linalg.cpu.nativecpu.buffer.BaseCpuDataBuffer;
import org.nd4j.linalg.cpu.nativecpu.buffer.LongBuffer;
import org.nd4j.linalg.cpu.nativecpu.buffer.Utf8Buffer;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Optional;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.*;

import java.util.*;

@Slf4j
public class NativeOpExecutioner extends DefaultOpExecutioner {
    private NativeOps loop =Nd4j.getNativeOps();
    private ConstantHandler constantHandler = Nd4j.getConstantHandler();
    @Getter
    private CpuTADManager tadManager = new CpuTADManager();

    protected Map<String, CustomOpDescriptor> customOps = null;

    protected ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();

    protected AtomicBoolean experimentalMode = new AtomicBoolean(false);


    public NativeOpExecutioner() {
        tadManager.init(loop, constantHandler);

        experimentalMode.set(loop.isExperimentalEnabled());

        // filling vars for possible overrides
        val env = System.getenv(ND4JEnvironmentVars.ND4J_MKL_FALLBACK);
        if (env != null) {
            // in this case we just disable mkl-dnn globally

        }
    }

    @Override
    public INDArray exec(Op op) {
        return exec(op, null);
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        checkForCompression(op);
        long start = profilingConfigurableHookIn(op,opContext);
        if (op instanceof ScalarOp) {
            ScalarOp s = (ScalarOp) op;
            exec(s, opContext);
        } else if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            exec(t, opContext);
        } else if (op instanceof ReduceOp) {
            ReduceOp ac = (ReduceOp) op;
            exec(ac, opContext);
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation iac = (IndexAccumulation) op;
            exec(iac, opContext); //Currently using DefaultOpExecutioner
        } else if (op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            exec(broadcastOp, opContext);
        } else if (op instanceof RandomOp) {
            RandomOp rngOp = (RandomOp) op;
            exec(rngOp, opContext, Nd4j.getRandom());
        }

        profilingConfigurableHookOut(op,opContext,start);
        return op.z();
    }


    @Override
    public INDArray exec(IndexAccumulation op) {
        return exec(op, null);
    }

    public INDArray exec(IndexAccumulation op, OpContext oc) {
        checkForCompression(op);

        INDArray x = getX(op, oc);
        INDArray z = getZ(op, oc);
        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val dimension = Shape.normalizeAxis(x.rank(), op.dimensions().toLongVector());

        if (x.isEmpty()) {
            for (val d:dimension) {
                Preconditions.checkArgument(x.size(d) != 0, "IndexReduce can't be issued along axis with 0 in shape");
            }
        }

        boolean keepDims = op.isKeepDims();
        long[] retShape = Shape.reductionShape(x, dimension, true, keepDims);

        if(z == null || x == z) {
            val ret = Nd4j.createUninitialized(DataType.INT64, retShape);

            setZ(ret, op, oc);
            z = ret;
        } else if(!Arrays.equals(retShape, z.shape())) {
            throw new IllegalStateException("Z array shape does not match expected return type for op " + op
                    + ": expected shape " + Arrays.toString(retShape) + ", z.shape()=" + Arrays.toString(z.shape()));
        }

        op.validateDataTypes();


        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(x, dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets);

        long st = profilingConfigurableHookIn(op, tadBuffers.getFirst());

        val xb = OpaqueNDArray.fromINDArray(x);
        val zb = OpaqueNDArray.fromINDArray(z);

        if (z.isScalar()) {
            loop.execIndexReduceScalar(dummy,op.opNum(),xb,zb,null);
        } else {
            OpaqueNDArray fromDims = OpaqueNDArray.fromINDArray(op.dimensions());
            loop.execIndexReduce(dummy,op.opNum(),xb,zb,fromDims, null);
        }

        if (loop.lastErrorCode() != 0) {
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            StringBuilder errorMessage = new StringBuilder();
            errorMessage.append("Op [").append(op.getClass().getSimpleName()).append("] execution failed\n");
            errorMessage.append("Inputs:\n");
            errorMessage.append("X:\n");
            errorMessage.append(x);
            errorMessage.append("\n");
            errorMessage.append("Z:\n");
            errorMessage.append(z);
            errorMessage.append("\n");
            errorMessage.append(loop.lastErrorMessage());
            errorMessage.append(differentialFunction.debugInfo());
            throw new RuntimeException(errorMessage.toString());
        }
        profilingConfigurableHookOut(op, oc, st);
        return getZ(op, oc);
    }

    @Override
    public INDArray exec(Variance op) {
        return exec((ReduceOp) op);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        return exec(op, null);
    }

    public INDArray exec(ReduceOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);
        Preconditions.checkNotNull(x, "Op.x() cannot be null: Was null for op %s", op);
        long st = profilingConfigurableHookIn(op, oc);
        op.validateDataTypes(oc);
        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()) {
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(z != null) {
                if(!x.isScalar() && !z.isScalar())
                    Preconditions.checkState(x.equalShapes(z), "For empty reductions, result (z) array must have same shape as x shape." +
                            " Got: x=%ndShape, z=%ndShape", x, z);
                //assign will crash if z < x. Just return empty z.
                if(z.length() < x.length())
                    return z;


                z.assign(x);
                return z;
            } else {
                setZ(x.dup(), op, oc);
                return z;
            }
        }

        val dimension = Shape.normalizeAxis(x.rank(), op.dimensions() != null ?  op.dimensions().toLongVector() : null);
        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        boolean keepDims = op.isKeepDims();
        long[] retShape = Shape.reductionShape(x, dimension, true, keepDims);


        if (x.isVector() && x.length() == ArrayUtil.prod(retShape) && ArrayUtil.prodLong(retShape) > 1 && y == null) {
            profilingConfigurableHookOut(op, oc, st);
            return op.noOp();
        }

        /**
         * This is the result array.
         * We create it only if we hadn't provided it before
         */
        INDArray ret;
        if (z == null || z == x) {
            if (op.isComplexAccumulation()) {
                long xT = x.tensorsAlongDimension(dimension);
                long yT = y.tensorsAlongDimension(dimension);

                ret = Nd4j.create(op.resultType(), new long[]{xT, yT});
            } else {
                if (y != null) {

                    //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
                    if(x.length() == y.length()) {
                        //Pairwise
                        if (x.tensorsAlongDimension(dimension) != y.tensorsAlongDimension(dimension)) {
                            throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                                    Arrays.toString(x.shape()) + ", y shape = " + Arrays.toString(y.shape()) +
                                    ", dimension = " + Arrays.toString(dimension) + ")");
                        }
                        //reduce ops can have second inputs as axes
                    } else if(!(op instanceof ReduceOp)) {
                        //Every X TAD vs. entirety of Y
                        val xTADSize = x.length() / x.tensorsAlongDimension(dimension);

                        if (xTADSize != y.length()) {
                            throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                                    " (x TAD size = " + xTADSize + ", y size = " + y.length());
                        }
                    }
                }

                DataType dt = oc != null ? op.resultType(oc) : op.resultType();
                ret = Nd4j.create(dt, retShape);

            }
            setZ(ret, op, oc);
            z = ret;
        } else {
            ret = z;
        }



        /**
         * Note because dimension arrays don't change,
         * we use an {@link ConstantHandler} which knows how to reserve memory
         * for immutable buffers for the dimensions.
         * This gives us a pointer which is passed around in libnd4j.
         */
        val xb = OpaqueNDArray.fromINDArray(x);
        val zb = OpaqueNDArray.fromINDArray(z);
        if (op instanceof Variance) {
            if (ret.isScalar()) {
                loop.execSummaryStatsScalar(null,op.opNum(),xb,zb,null,((Variance) op).isBiasCorrected());

            } else {
                try {

                    loop.execSummaryStatsTad(
                            null,op.
                                    opNum(),
                            xb,
                            zb,
                            OpaqueNDArray.fromINDArray(op.dimensions())
                            ,null,
                            ((Variance) op).isBiasCorrected());

                } catch (Throwable t) {
                    String str = opInfoString(op, Optional.of(dimension));
                    StringBuilder errorMessage = new StringBuilder();
                    DifferentialFunction differentialFunction = (DifferentialFunction) op;
                    errorMessage.append("Native AccumulationOp execution (double) failed: " + str +  t);
                    errorMessage.append(differentialFunction.debugInfo());
                    throw new RuntimeException(errorMessage.toString());
                }
            }

        }
        //pairwise reduction like similarity of two arrays
        else if (y != null && op.getOpType() == Op.Type.REDUCE3) {
            val yb = OpaqueNDArray.fromINDArray(y);
            if (op.isComplexAccumulation()) {
                try {
                    //use opaque ndarrays instead here
                    loop.execReduce3All(null,
                            op.opNum(),
                            xb,yb,zb,OpaqueNDArray.fromINDArray(op.dimensions()),null);
                } catch (Throwable t) {
                    String str = opInfoString(op, Optional.of(dimension));
                    StringBuilder errorMessage = new StringBuilder();
                    DifferentialFunction differentialFunction = (DifferentialFunction) op;
                    errorMessage.append("Native AccumulationOp execution (double) failed: " + str +  t);
                    errorMessage.append(differentialFunction.debugInfo());
                    throw new RuntimeException(errorMessage.toString());
                }
            } else if (ret.isScalar()) {
                loop.execReduce3Scalar(null,op.opNum(),xb,null,yb,zb);
            } else {
                try {
                    loop.execReduce3Tad(null,op.opNum(),xb,null, yb, zb, OpaqueNDArray.fromINDArray(op.dimensions()));

                } catch (Throwable t) {
                    String str = opInfoString(op, Optional.of(dimension));
                    StringBuilder errorMessage = new StringBuilder();
                    DifferentialFunction differentialFunction = (DifferentialFunction) op;
                    errorMessage.append("Native AccumulationOp execution (double) failed: " + str +  t);
                    errorMessage.append(differentialFunction.debugInfo());
                    throw new RuntimeException(errorMessage.toString());
                }
            }

        } else {if (extraz.get() == null)
            extraz.set(new PointerPointer(32));
            OpaqueNDArray dims = OpaqueNDArray.fromINDArray(op.dimensions());

            if (ret.isScalar()) {
                if (extraz.get() == null)
                    extraz.set(new PointerPointer(32));
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        loop.execReduceFloat(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb);
                        break;
                    case REDUCE_BOOL:
                        loop.execReduceBool(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb,dims);
                        break;
                    case REDUCE_SAME:
                        loop.execReduceSame(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb);
                        break;
                    case REDUCE_LONG:
                        loop.execReduceLong(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb,dims);
                        break;
                    default:
                        throw new UnsupportedOperationException("Unsupported op used in reduce: " + op.getOpType());
                }
            } else {
                if (extraz.get() == null)
                    extraz.set(new PointerPointer(32));
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        loop.execReduceFloat2(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()), zb,OpaqueNDArray.fromINDArray(op.dimensions()));
                        break;
                    case REDUCE_LONG:
                        loop.execReduceLong2(null, op.opNum(), xb,getPointerForExtraArgs(op, x.dataType()), zb, OpaqueNDArray.fromINDArray(op.dimensions()));
                        break;
                    case REDUCE_SAME:
                        loop.execReduceSame2(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb, OpaqueNDArray.fromINDArray(op.dimensions()));
                        break;
                    case REDUCE_BOOL:
                        loop.execReduceBool2(null, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb, OpaqueNDArray.fromINDArray(op.dimensions()));
                        break;

                    default:
                        throw new UnsupportedOperationException("Unsupported op used in reduce: " + op.getOpType());
                }
            }
        }

        if (loop.lastErrorCode() != 0) {
            String str = opInfoString(op, Optional.of(dimension));
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native AccumulationOp execution (double) failed: " + str);
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(loop.lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
        profilingConfigurableHookOut(op, oc, st);
        return getZ(op, oc);
    }

    /**
     * ScalarOp execution
     * @param op Op to execute
     */
    private void invokeScalarAlongDimension(ScalarOp op) {
        invokeScalarAlongDimension(op, null);
    }

    private void invokeScalarAlongDimension(ScalarOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);
        val dimension = op.dimensions().toLongVector();

        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);
        switch (op.getOpType()) {
            case SCALAR:
                loop.execScalarTad(null, op.opNum(),
                        xb,
                        zb,
                        yb,
                        getPointerForExtraArgs(op, x.dataType()),
                        OpaqueNDArray.fromINDArray(op.dimensions())
                );
                break;
            case SCALAR_BOOL:
                loop.execScalarTad(null, op.opNum(),
                        xb,
                        zb,
                        yb,
                        getPointerForExtraArgs(op, x.dataType()),
                        OpaqueNDArray.fromINDArray(op.dimensions())
                );
                break;
            default:
                throw new UnsupportedOperationException();
        }

        if (loop.lastErrorCode() != 0) {
            String str = opInfoString(op, Optional.of(dimension));
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native  execution exec failed: " + str);
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(loop.lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
    }

    public INDArray exec(ScalarOp op) {
        return exec(op, null);
    }

    public INDArray exec(ScalarOp op, OpContext oc) {
        long st = profilingConfigurableHookIn(op);
        if((oc != null && oc.getOutputArray(0) == null) || getZ(op, oc) == null) {
            switch (op.getOpType()) {
                case SCALAR:
                    setZ(getX(op, oc).ulike(), op, oc);
                    break;
                case SCALAR_BOOL:
                    setZ(Nd4j.createUninitialized(DataType.BOOL, getX(op, oc).shape()), op, oc);
                    break;
                default:
                    throw new ND4JIllegalStateException("Unknown op type: [" + op.getOpType() +"]");
            }
        }


        if (op.dimensions() != null) {
            invokeScalarAlongDimension(op);
            return getZ(op, oc);
        }

        val x = OpaqueNDArray.fromINDArray(getX(op, oc));
        val scalar = OpaqueNDArray.fromINDArray(op.scalar());
        val z =  OpaqueNDArray.fromINDArray(getZ(op, oc));


        switch (op.getOpType()) {
            case SCALAR:
                loop.execScalar(null,op.opNum(),x,z,scalar,getPointerForExtraArgs(op, x.dataType()));
                break;
            case SCALAR_BOOL:
                loop.execScalarBool(null,op.opNum(),x,z,scalar,getPointerForExtraArgs(op, x.dataType()));
                break;
            default:
                throw new ND4JIllegalStateException("Unknown op type: [" + op.getOpType() +"]");
        }

        if (loop.lastErrorCode() != 0) {
            // the variable is mainly for ease of use with the debugger
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native  execution exec failed: ");
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(loop.lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
        profilingConfigurableHookOut(op, oc, st);
        return getZ(op, oc);
    }

    private Pointer getPointerForExtraArgs(Op op, DataType type) {
        if (op.extraArgs() != null) {
            val eadb = op.extraArgsDataBuff(type);
            if (eadb != null)
                return eadb.addressPointer();
            else
                return null;
        }

        return null;
    }


    private void exec(TransformOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);
        long st = profilingConfigurableHookIn(op,oc);
        //redirect assign so we support more ops cases lke strings
        if(op instanceof Assign) {
            DefaultOpExecutioner.execAssign(op, oc,this);
        } else {
            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            PointerPointer dummy = extraz.get();

            // Pow operations might be special
            if (op.opNum() == 31) {
                if (y != null && y.isScalar()) {
                    setY(Nd4j.valueArrayOf(x.shape(), y.getDouble(0)), op, oc);
                }
            }

            /**
             * This is the {@link IsMax}
             * operation.
             *
             * @see {@link Op#extraArgs()}
             * for what an extra argument is in an op.
             *
             * The extra argument in the op here is the {@link IsMax#IsMax(INDArray, int...)}
             * dimension to do the ismax along
             */
            if (op.opName().equalsIgnoreCase("ismax") && op.extraArgs() != null && op.extraArgs().length > 0) {
                long[] dimension = new long[(int) op.extraArgs()[0]];

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

                st = profilingConfigurableHookIn(op, tadBuffers.getFirst());
            } else
                st = profilingConfigurableHookIn(op);

            if (y != null) {

                if (z == null) {
                    setZ(Nd4j.create(op.resultType(), x.shape()), op, oc);
                    z = getZ(op, oc);
                }


                op.validateDataTypes(oc, experimentalMode.get());



                val xb =  OpaqueNDArray.fromINDArray(x);
                val yb = OpaqueNDArray.fromINDArray(y);
                val zb = OpaqueNDArray.fromINDArray(z);
                ((BaseCpuDataBuffer) x.data()).actualizePointerAndIndexer();
                ((BaseCpuDataBuffer) z.data()).actualizePointerAndIndexer();
                switch (op.getOpType()) {
                    case TRANSFORM_ANY:
                    case TRANSFORM_FLOAT:
                    case TRANSFORM_STRICT:
                    case TRANSFORM_SAME:
                        loop.execPairwiseTransform(dummy,op.opNum(),xb,yb,zb, getPointerForExtraArgs(op, x.dataType()));
                        break;
                    case TRANSFORM_BOOL:
                        loop.execTransformBool(dummy, op.opNum(), xb, getPointerForExtraArgs(op, x.dataType()),zb);
                        break;
                    case PAIRWISE_BOOL:
                        loop.execPairwiseTransformBool(dummy, op.opNum(), xb, yb, getPointerForExtraArgs(op, x.dataType()),zb);
                        break;
                }
            } else {

                if (z == null) {
                    setZ(Nd4j.createUninitialized((oc != null ? op.resultType(oc) : op.resultType()), x.shape()), op, oc);
                    z = getZ(op, oc);
                }

                op.validateDataTypes(oc, experimentalMode.get());

                val xb = OpaqueNDArray.fromINDArray(x);
                val zb = OpaqueNDArray.fromINDArray(z);

                if (extraz.get() == null)
                    extraz.set(new PointerPointer(32));

                switch (op.getOpType()) {
                    case TRANSFORM_FLOAT: {
                        val xtraz = getPointerForExtraArgs(op, z.dataType());
                        loop.execTransformFloat(dummy, op.opNum(), xb,xtraz, zb);
                        break;
                    }
                    case TRANSFORM_STRICT: {
                        val xtraz = getPointerForExtraArgs(op, z.dataType());
                        loop.execTransformStrict(dummy, op.opNum(), xb, xtraz,zb);
                        break;
                    }
                    case TRANSFORM_SAME: {
                        val xtraz = getPointerForExtraArgs(op, z.dataType());
                        loop.execTransformSame(dummy, op.opNum(), xb,xtraz, zb);
                        break;
                    }


                    default:
                        throw new UnsupportedOperationException("Unknown transform type: [" + op.getOpType() + "]");
                }

            }

            if (loop.lastErrorCode() != 0) {
                StringBuilder errorMessage = new StringBuilder();
                DifferentialFunction differentialFunction = (DifferentialFunction) op;
                errorMessage.append("Native  execution exec failed: ");
                errorMessage.append(differentialFunction.debugInfo());
                errorMessage.append(loop.lastErrorMessage());
                throw new RuntimeException(errorMessage.toString());
            }
        }


        profilingConfigurableHookOut(op, oc, st);
    }

    public INDArray exec(BroadcastOp op) {
        return exec(op, null);
    }

    public INDArray exec(BroadcastOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);
        long st = profilingConfigurableHookIn(op,oc);
        op.validateDataTypes(experimentalMode.get());

        val dimension = op.dimensions().toLongVector();

        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(x, dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(z, dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, devTadShapeInfoZ, devTadOffsetsZ);


        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);
        OpaqueNDArray dimArray = OpaqueNDArray.fromINDArray(op.dimensions());
        switch (op.getOpType()) {
            case BROADCAST:
                loop.execBroadcast(dummy,op.opNum(),xb, yb, zb,extraz.get(),dimArray);

                break;
            case BROADCAST_BOOL:
                loop.execBroadcastBool(dummy,op.opNum(),xb, yb,zb,extraz.get(),dimArray);
                break;
            default:
                throw new UnsupportedOperationException("Unknown operation type: [" + op.getOpType() + "]");
        }

        if (loop.lastErrorCode() != 0) {
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native  execution exec failed: ");
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(loop.lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
        profilingConfigurableHookOut(op,oc,st);
        return z;
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
        return exec(op, null, rng);
    }


    public INDArray exec(RandomOp op, OpContext oc, Random rng) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        if(op instanceof BaseRandomOp && ((BaseRandomOp)op).isTripleArgRngOp() && z != null && x == null && y == null) {
            //Ugly hack to ensure the triple arg call occurs
            //See GaussianDistribution.setZ etc
            x = z;
            y = z;
        }

        if (!(rng instanceof CpuNativeRandom))
            throw new IllegalStateException(
                    "You should use one of NativeRandom classes for NativeOperations execution. Op class: " + op.getClass().getName());



        if(z != null)
            Preconditions.checkArgument(z.isR(), "Op.Z must have one of floating point types");

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);

        if (x != null && y != null && z != null) {
            DataBuffer dataBuffer = op.extraArgsDataBuff(z.dataType());
            loop.execRandom3(null,op.opNum(),rng.getStatePointer(),xb,yb,zb,dataBuffer.addressPointer());
        } else if (x != null && z != null) {
            DataBuffer dataBuffer = op.extraArgsDataBuff(z.dataType());
            loop.execRandom2(null,op.opNum(),rng.getStatePointer(),xb,zb,dataBuffer.addressPointer());
        } else {
            DataBuffer dataBuffer = op.extraArgsDataBuff(z.dataType());
            loop.execRandom(null,op.opNum(),rng.getStatePointer(),zb,dataBuffer.addressPointer());
        }

        if (loop.lastErrorCode() != 0) {
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native  execution exec failed: ");
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(loop.lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }

        return z;
    }

    @Override
    public TADManager getTADManager() {
        return tadManager;
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




    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * @param op Operation to execute
     */
    @Override
    public  INDArray[] exec(@NonNull CustomOp op) {
        boolean shapeOverride = op.initializeOutputs(null);
        val name = op.opName();
        try (val context = buildContext()) {
            long start = profilingConfigurableHookIn(op,context);
            initOpContext(op, shapeOverride, context);

            val result = exec(op, context);
            val states = context.getRngStates();


            // pulling states back
            Nd4j.getRandom().setStates(states.getFirst(), states.getSecond());
            profilingConfigurableHookOut(op,context,start);

            return result;
        } catch (ND4JOpProfilerException e) {

            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Op [" + name + "] execution failed", e);
        }


    }

    protected LongShapeDescriptor getShapeFromPointer(LongPointer ptr) {
        val rank = (int) ptr.get(0);

        val shape = new long[rank * 2 + 4];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = ptr.get(i);
        }

        LongShapeDescriptor ret = LongShapeDescriptor.builder()
                .shape(Shape.shape(shape))
                .stride(Shape.stride(shape))
                .order(Shape.order(shape))
                .ews(Shape.elementWiseStride(shape))
                .extras(Shape.extras(shape))
                .build();
        return ret;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(@NonNull CustomOp op) {
        return calculateOutputShape(op, null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(@NonNull CustomOp op, OpContext opContext) {
        DifferentialFunction func = (DifferentialFunction) op;
        String opName = func.getOwnName();
        val lc = op.opName().toLowerCase();
        val hash = op.opHash();

        val result = new ArrayList<LongShapeDescriptor>();
        int nIn = opContext != null ? opContext.numInputArguments() : op.numInputArguments();
        if (nIn == 0 && op.getDescriptor().getNumInputs() >= 1) {
            if (log.isTraceEnabled()) {
                log.trace("Could not calculate output shape for op {}: number of input args was 0",
                        op.getClass().getName());
            }
            return Collections.emptyList();
        }

        val inputBuffers = new PointerPointer<>(nIn);
        val inputShapes = new PointerPointer<>(nIn);
        val inputArgs = opContext != null && opContext.getInputArrays() != null && !opContext.getInputArrays().isEmpty()
                ? opContext.getInputArrays() : op.inputArguments();
        int cnt = 0;
        int numProcessed = 0;
        long[] offsets = new long[nIn];
        for (val in : inputArgs) {
            if (!in.isEmpty())
                inputBuffers.put(cnt, in.data().addressPointer());
            offsets[cnt] = in.offset();

            inputShapes.put(cnt++, in.shapeInfoDataBuffer().addressPointer());
            numProcessed++;
        }

        if (numProcessed != nIn) {
            throw new ND4JIllegalStateException("Number of processed inputs should match number of inputs. " +
                    "Got " + numProcessed + " inputs but should have been " + nIn + " . This is likely due a null input.");
        }

        int nIArgs = opContext != null ? opContext.numIArguments() : op.numIArguments();
        val iArgs = nIArgs > 0 ? new LongPointer(nIArgs) : null;
        cnt = 0;
        if (opContext != null) {
            if (iArgs != null)
                for (val i : opContext.getIArguments())
                    iArgs.put(cnt++, i);
        } else {
            if (iArgs != null)
                for (val i : op.iArgs())
                    iArgs.put(cnt++, i);
        }

        int nTArgs = opContext != null ? opContext.numTArguments() : op.numTArguments();
        val tArgs = nTArgs > 0 ? new DoublePointer(nTArgs) : null;

        int nBArgs = opContext != null ? opContext.numBArguments() : op.numBArguments();
        val bArgs = nBArgs > 0 ? new BooleanPointer(nBArgs) : null;

        int nDArgs = opContext != null ? opContext.numDArguments() : op.numDArguments();
        val dArgs = nDArgs > 0 ? new IntPointer(nDArgs) : null;

        cnt = 0;
        if (opContext != null) {
            if (bArgs != null)
                for (val b : opContext.getBArguments())
                    bArgs.put(cnt++, b);
        } else {
            if (bArgs != null)
                for (val b : op.bArgs())
                    bArgs.put(cnt++, b);
        }

        cnt = 0;
        if (opContext != null) {
            if (tArgs != null)
                for (val b : opContext.getTArguments())
                    tArgs.put(cnt++, b);
        } else {
            if (tArgs != null)
                for (val b : op.tArgs())
                    tArgs.put(cnt++, b);
        }

        cnt = 0;
        if (opContext != null) {
            if (dArgs != null)
                for (val b : opContext.getDArguments())
                    dArgs.put(cnt++, b.toInt());
        } else {
            if (dArgs != null)
                for (val b : op.dArgs())
                    dArgs.put(cnt++, b.toInt());
        }

        if (numProcessed != nIn) {
            throw new ND4JIllegalStateException("Number of processed inputs should match number of inputs. " +
                    "Got " + numProcessed + " inputs but should have been " + nIn + " . This is likely due a null input.");
        }

        OpaqueShapeList ptrptr;
        try {
            OpaqueNDArrayArr inputsOpaque = new OpaqueNDArrayArr(inputArgs.stream().map(OpaqueNDArray::fromINDArray).toArray(OpaqueNDArray[]::new));
            ptrptr = loop.calculateOutputShapes2(null, hash, inputsOpaque, nIn, tArgs, nTArgs, iArgs, nIArgs, bArgs, nBArgs, dArgs, nDArgs);

            if (loop.lastErrorCode() != 0) {
                StringBuilder errorMessage = new StringBuilder();
                DifferentialFunction differentialFunction = (DifferentialFunction) op;
                errorMessage.append("Native execution exec failed: ");
                errorMessage.append(differentialFunction.debugInfo());
                errorMessage.append(loop.lastErrorMessage());
                throw new RuntimeException(errorMessage.toString());
            }
            if (ptrptr == null)
                throw new RuntimeException();

        } catch (Throwable t) {
            StringBuilder sb = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            sb.append("Inputs: [(");
            for (int i = 0; i < inputArgs.size(); i++) {
                if (i > 0)
                    sb.append("), (");
                sb.append(Shape.shapeToStringShort(inputArgs.get(i)));
            }
            sb.append(")]");
            sb.append(differentialFunction.debugInfo());

            int nOut = opContext != null ? opContext.numOutputArguments() : op.numOutputArguments();
            log.error("Failed to calculate output shapes for op {}. Attempted to execute with {} inputs, {} outputs, " +
                            "{} targs, {} iargs, {} bargs and {} dargs. {} - Please see above message (printed out from c++) for a possible cause of error.",
                    op.opName(), nIn, nOut, nTArgs, nIArgs, nBArgs, nDArgs, sb.toString());
            throw t;
        }

        if (loop.lastErrorCode() != 0) {
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native execution exec failed: ");
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(loop.lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
        if (ptrptr == null)
            throw new RuntimeException();

        for (int e = 0; e < loop.getShapeListSize(ptrptr); e++)
            result.add(getShapeFromPointer(new PagedPointer(loop.getShape(ptrptr, e)).asLongPointer()));

        if (log.isTraceEnabled()) {
            String[] arr = new String[result.size()];
            for (int i = 0; i < result.size(); i++) {
                arr[i] = result.get(i).toString();
            }

            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            log.trace("Calculated output shapes for op of name {} and type {} - {}", differentialFunction.getOwnName(), op.getClass().getName(), Arrays.toString(arr));
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

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
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

        OpaqueVariablesSet result = loop.executeStoredGraph(null, id, ptrBuffers, ptrShapes, ptrIndices, map.size());

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        OpStatus status = OpStatus.byNumber(loop.getVariablesSetStatus(result));

        if (status != OpStatus.ND4J_STATUS_OK)
            throw new ND4JIllegalStateException("Op execution failed: " + status);

        for (int e = 0; e < loop.getVariablesSetSize(result); e++) {
            OpaqueVariable var = loop.getVariable(result, e);
            int nodeId = loop.getVariableId(var);
            int index = loop.getVariableIndex(var);
            LongPointer shapeInfo = loop.getVariableShape(var);
            Pointer buffer = loop.getVariableBuffer(var);

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

            Pointer.memcpy(array.data().addressPointer(), buffer, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(array.dataType()));

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(array.dataType()), MemcpyDirection.HOST_TO_HOST);

            String nodeName = loop.getVariableName(var);
            newMap.put(nodeName, array);
        }

        // loop.deleteVariablesSet(result);

        return newMap;
    }

    @Override
    public void forgetGraph(long id) {
        loop.unregisterGraph(null, id);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
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

    @Override
    public String getString(DataBuffer buffer, long index) {
        Preconditions.checkArgument(buffer instanceof Utf8Buffer, "Expected Utf8Buffer");

        val addr = ((LongIndexer) buffer.indexer()).get(index);
        val ptr = new PagedPointer(addr);
        return "";
    }

    @Override
    public ExecutionerType type() {
        return ExecutionerType.NATIVE_CPU;
    }

    @Override
    public boolean isExperimentalMode() {
        return experimentalMode.get();
    }

    @Override
    public void scatterUpdate(ScatterUpdate.UpdateOp op, @NonNull INDArray array, @NonNull INDArray indices, @NonNull INDArray updates, long[] axis) {
        val tadX = tadManager.getTADOnlyShapeInfo(array, axis);
        val tadY = tadManager.getTADOnlyShapeInfo(updates, axis);

        if (tadY.getSecond().length() != indices.length())
            throw new IllegalStateException("Number of updates doesn't match number of indices. Bad dimensions used?");

        val arrayOpaque = OpaqueNDArray.fromINDArray(array);
        val updatesOpaque = OpaqueNDArray.fromINDArray(updates);
        val indicesOpaque = OpaqueNDArray.fromINDArray(indices);

        INDArray dimm = Nd4j.createFromArray(axis);
        val dimmOpaque = OpaqueNDArray.fromINDArray(dimm);

        loop.scatterUpdate(null,op.ordinal(),arrayOpaque,indicesOpaque,updatesOpaque,dimmOpaque);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
    }

    @Override
    public OpContext buildContext() {
        if(this.nextOpContext.get() != null) {
            return this.nextOpContext.get();
        }

        CpuOpContext ctx =  new CpuOpContext();
        return ctx;
    }

    @Override
    public INDArray[] exec(CustomOp op, @NonNull OpContext context) {
        long st = profilingConfigurableHookIn(op, context);


        try {

            if(op instanceof UserDefinedCustomOp) {
                ((UserDefinedCustomOp) op).exec(context);
                return context.getOutputArrays().toArray(new INDArray[0]);
            }



            val status = loop.execCustomOp2(null, op.opHash(), context.contextPointer());


            if (status != 0) {
                StringBuilder errorMessage = new StringBuilder();
                DifferentialFunction differentialFunction = (DifferentialFunction) op;
                errorMessage.append("Native  execution exec failed: ");
                errorMessage.append(differentialFunction.debugInfo());
                errorMessage.append(loop.lastErrorMessage());
                throw new RuntimeException(errorMessage.toString());
            }
            if (context.getOutputArrays().isEmpty())
                return new INDArray[0];
            else
                return context.getOutputArrays().toArray(new INDArray[context.getOutputArrays().size()]);
        } catch (Exception e) {
            val sb = new StringBuilder();
            sb.append("Inputs: [(");
            int nIn = (context.getInputArrays() == null ? 0 : context.getInputArrays().size());
            for (int i = 0; i < nIn; i++) {
                if (i > 0)
                    sb.append("), (");
                sb.append(Shape.shapeToStringShort(context.getInputArrays().get(i)));
            }
            sb.append(")]. Outputs: [(");
            int nOut = (context.getOutputArrays() == null ? 0 : context.getOutputArrays().size());
            for (int i = 0; i < nOut; i++) {
                if (i > 0)
                    sb.append("), (");
                sb.append(Shape.shapeToStringShort(context.getOutputArrays().get(i)));
            }
            sb.append(")]. tArgs: ");
            int nT = (context.getTArguments() == null ? 0 : context.getTArguments().size());
            if (nT > 0) {
                sb.append(context.getTArguments());
            } else {
                sb.append("-");
            }
            sb.append(". iArgs: ");
            int nI = (context.getIArguments() == null ? 0 : context.getIArguments().size());
            if (nI > 0) {
                sb.append(context.getIArguments());
            } else {
                sb.append("-");
            }
            sb.append(". bArgs: ");
            int nB = (context.getBArguments() == null ? 0 : context.getBArguments().size());
            if (nB > 0) {
                sb.append(context.getBArguments());
            } else {
                sb.append("-");
            }
            if (op instanceof DifferentialFunction) {
                String n = ((DifferentialFunction) op).getOwnName();
                if (n != null && !n.equals(op.opName())) {
                    sb.append(". Op own name: \"").append(n).append("\"");
                }
            }

            if(op instanceof DifferentialFunction && ((DifferentialFunction)op).getSameDiff() != null) {
                appendSameDiffInfo(sb, (DifferentialFunction) op);
            }

            log.error("Failed to execute op " + op.opName() + ". Attempted to execute with " +
                    nIn + " inputs, " +
                    nOut + " outputs, " +
                    nT + " targs," +
                    nB + " bargs and " +
                    nI + " iargs. " +
                    sb.toString() +
                    " - Please see above message (printed out from c++) for a possible cause of error.");
            throw e;
        } finally {
            profilingConfigurableHookOut(op, context, st);
        }
    }

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        return Nd4j.getStatsProvider().inspectArray(array);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty, boolean isView) {
        long[] merged = new long[Shape.shapeInfoLength(shape.length)];

        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            DataBuffer ret = Nd4j.createBuffer(DataType.INT64,Shape.shapeInfoLength(shape.length),true);
            merged[0] = shape.length;
            int shapeIdx = 0;
            int strideIdx = 0;
            for(int i = 1; i < shape.length * 2 + 1; i++) {
                if(shapeIdx < shape.length) {
                    merged[i] = shape[shapeIdx];
                    shapeIdx++;
                } else {
                    merged[i] = stride[strideIdx];
                    strideIdx++;
                }
            }



            Shape.setElementWiseStride(merged,(int) elementWiseStride);
            LongPointer longPointer = new LongPointer(merged);
            loop.setShapeBuffer(longPointer,dtype.toInt(),new LongPointer(ret.addressPointer()),order,(int) elementWiseStride,empty,isView);
            longPointer.deallocate();
            longPointer.releaseReference();
            if(isView != ArrayOptionsHelper.isView(Shape.options(ret))) {
                throw new IllegalStateException("isView is not set properly");
            }

            if(empty != ArrayOptionsHelper.isEmpty(Shape.options(ret))) {
                throw new IllegalStateException("Empty is not set properly");
            }


            long[] shape2 = Shape.shape(ret.asLong());
            long[] stride2 = Shape.stride(ret.asLong());
            long ews = Shape.elementWiseStride(ret.asLong());
            char order2 = Shape.order(ret.asLong());
            DataType dtype2 = ArrayOptionsHelper.dataType(Shape.options(ret));
            boolean empty2 = ArrayOptionsHelper.isEmpty(Shape.options(ret));
            boolean isView2 = ArrayOptionsHelper.isView(Shape.options(ret));
            if(!Arrays.equals(shape,shape2)) {
                throw new IllegalStateException("Shape is not set properly");
            }

            if(!Arrays.equals(stride,stride2)) {
                throw new IllegalStateException("Stride is not set properly");
            }

            if(ews > 0 && ews != elementWiseStride) {
                throw new IllegalStateException("Element wise stride is not set properly");
            }

            if(order != order2) {
                throw new IllegalStateException("Order is not set properly");
            }

            if(dtype != dtype2) {
                throw new IllegalStateException("Data type is not set properly");
            }

            if(empty != empty2) {
                throw new IllegalStateException("Empty is not set properly");
            }

            if(isView != isView2) {
                throw new IllegalStateException("Is view is not set properly");
            }
            return ret;
        }


    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        return createShapeInfo(shape,stride,elementWiseStride,order,dtype,empty,false);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extras) {
        LongPointer shapePointer = new LongPointer(shape);
        LongPointer stridePointer = new LongPointer(stride);
        OpaqueConstantShapeBuffer dbf = loop.shapeBufferEx(shape.length, shapePointer, stridePointer, dtype.toInt(), order, elementWiseStride, extras);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        val result = new LongBuffer(loop.getConstantShapeBufferPrimary(dbf), Shape.shapeInfoLength(shape.length));

        return result;
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension) {
        long[] inputDimensions = new long[dimension.length];
        for(int i = 0; i < inputDimensions.length; i++) {
            inputDimensions[i] = dimension[i];
        }
        try {
            OpaqueTadPack pack = loop.tadOnlyShapeInfo(array.shapeInfoDataBuffer().opaqueBuffer(), new LongPointer(inputDimensions), dimension.length);

            if (loop.lastErrorCode() != 0)
                throw new RuntimeException(loop.lastErrorMessage());

            val tadShape = new LongBuffer(loop.getPrimaryShapeInfo(pack), loop.getShapeInfoLength(pack));
            val tadOffsets = new LongBuffer(loop.getPrimaryOffsets(pack), loop.getNumberOfTads(pack));
            return new TadPack(tadShape, tadOffsets);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

    protected void appendSameDiffInfo(StringBuilder sb, DifferentialFunction df) {
        String[] inNames = df.argNames();
        String[] outNames = df.outputVariablesNames();
        if(inNames != null){
            sb.append(". Input var names: ").append(Arrays.toString(inNames));
        }
        if(outNames != null) {
            sb.append(". Output var names: ").append(Arrays.toString(outNames));
        }
    }

    @Override
    public int useCount(DataBuffer buffer) {
        return loop.dbUseCount(((BaseCpuDataBuffer) buffer).getOpaqueDataBuffer());
    }



}
