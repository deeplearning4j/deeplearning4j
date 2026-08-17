/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan.ops.executioner;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.ops.ReduceOp;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.vulkan.VulkanAffinityManager;
import org.nd4j.linalg.vulkan.VulkanDataBuffer;
import org.nd4j.linalg.vulkan.VulkanDataBufferFactory;
import org.nd4j.linalg.vulkan.VulkanNDArray;
import org.nd4j.linalg.vulkan.VulkanRuntime;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueConstantDataBuffer;
import org.nd4j.nativeblas.OpaqueConstantShapeBuffer;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.nativeblas.OpaqueShapeList;
import org.nd4j.nativeblas.OpaqueTadPack;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Vulkan execution service.
 *
 * <p>Graph and eager execution are admitted by the same descriptor/hash-driven
 * Vulkan catalog. Java owns context construction and coherence; native Vulkan
 * owns kernel selection, pipeline dispatch, and unsupported-op rejection.</p>
 */
@Slf4j
public final class VulkanExecutioner extends DefaultOpExecutioner {
    private final TADManager tadManager;
    private final Nd4jVulkan nativeOps;
    private final VulkanAffinityManager affinityManager;
    private final VulkanDataBufferFactory dataBufferFactory;
    private volatile Map<String, CustomOpDescriptor> customOps;
    private volatile Map<Long, CustomOpDescriptor> customOpsByHash;

    public VulkanExecutioner() {
        this(VulkanRuntime.getInstance().nativeOps(),
                VulkanRuntime.getInstance().affinityManager(),
                VulkanRuntime.getInstance().dataBufferFactory());
    }

    public VulkanExecutioner(NativeOps nativeOps, VulkanAffinityManager affinityManager,
                             VulkanDataBufferFactory dataBufferFactory) {
        if (!(nativeOps instanceof Nd4jVulkan)
                || affinityManager == null || dataBufferFactory == null) {
            throw new IllegalArgumentException("Vulkan execution services must not be null");
        }
        this.nativeOps = (Nd4jVulkan) nativeOps;
        this.affinityManager = affinityManager;
        this.dataBufferFactory = dataBufferFactory;
        this.tadManager = new VulkanTADManager(this);
    }

    private void checkNativeError() {
        int errorCode = nativeOps.lastErrorCode();
        if (errorCode != 0) {
            String message = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "Vulkan native error " + errorCode + ": " + message);
        }
    }

    void purgeTadCache() {
        checkNativeError();
        nativeOps.clearTADCache();
        checkNativeError();
    }

    long tadCachedBytes() {
        checkNativeError();
        long bytes = nativeOps.getTADCachedBytes();
        checkNativeError();
        return bytes;
    }

    @Override
    public INDArray exec(@NonNull Op op) {
        return exec(op, null);
    }

    @Override
    public INDArray exec(@NonNull Op op, OpContext opContext) {
        if (op instanceof TransformOp) {
            executeTransform((TransformOp) op, opContext);
        } else if (op instanceof ReduceOp) {
            executeReduction((ReduceOp) op, opContext);
        } else if (op instanceof ScalarOp) {
            executeScalar((ScalarOp) op, opContext);
        } else if (op instanceof RandomOp) {
            executeRandom((RandomOp) op, opContext, Nd4j.getRandom());
        } else {
            /*
             * Legacy Op is already able to describe itself as a DynamicCustomOp.
             * Consult the native descriptor registry before choosing a legacy
             * family-specific entry point. This keeps typed execution extensible:
             * every descriptor/emitter that is admitted by the Vulkan catalog can
             * execute from eager ND4J APIs without a Java allow-list.
             */
            CustomOp customOp = op.toCustomOp();
            // The descriptor hash is the single source of truth. Do not maintain a
            // second Java operation allow-list: native catalog validation selects
            // the emitter and reports a missing lowering with the canonical hash.
            executeCustom(customOp, null);
        }
        return getZ(op, opContext);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        return exec((Op) op);
    }

    @Override
    public INDArray exec(Variance op) {
        return exec((Op) op);
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        return exec((Op) op);
    }

    @Override
    public INDArray exec(BroadcastOp op) {
        return exec((Op) op);
    }

    @Override
    public INDArray exec(ScalarOp op) {
        return exec((Op) op);
    }

    private void executeScalar(ScalarOp op, OpContext opContext) {
        INDArray x = getX(op, opContext);
        if (x == null) {
            throw new IllegalArgumentException("Vulkan scalar execution requires an input array");
        }

        int deviceId = affinityManager.getDeviceForArray(x);
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        try {
            if (previousDevice != deviceId) {
                affinityManager.setDeviceForCurrentThread(deviceId);
            }

            INDArray scalar = op.scalar();
            if (scalar == null) {
                scalar = getY(op, opContext);
            }
            if (scalar == null) {
                throw new IllegalArgumentException(
                        "Vulkan scalar execution requires a scalar operand");
            }
            if (affinityManager.getDeviceForArray(scalar) != deviceId) {
                scalar = affinityManager.replicateToDevice(deviceId, scalar);
            }
            if (scalar.dataType() != x.dataType()) {
                scalar = scalar.castTo(x.dataType());
            }

            INDArray z = getZ(op, opContext);
            if (z == null) {
                switch (op.getOpType()) {
                    case SCALAR:
                        z = x.ulike();
                        break;
                    case SCALAR_BOOL:
                        z = Nd4j.createUninitialized(DataType.BOOL, x.shape());
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                "Unknown Vulkan scalar family " + op.getOpType());
                }
                setZ(z, op, opContext);
            }

            requireSameDevice(z, deviceId, "output");
            op.validateDataTypes(nativeOps.isExperimentalEnabled());
            DataType extraType = op.getOpType() == Op.Type.SCALAR_BOOL
                    ? x.dataType() : z.dataType();
            Object[] extraArgs = op.extraArgs();
            DataBuffer extraBuffer = extraArgs == null || extraArgs.length <= 1
                    ? null : op.extraArgsDataBuff(extraType);
            // LegacyScalarOp turns TArg zero into the explicit scalar tensor and
            // forwards only the remaining values as kernel extra parameters.
            Pointer extraArguments = extraBuffer == null
                    ? null
                    : new Pointer(extraBuffer.addressPointer()).position(extraType.width());
            INDArray dimensions = op.dimensions();
            VulkanRuntime runtime = VulkanRuntime.forNativeOps(nativeOps);

            nativeOps.clearLastError();
            try (OpaqueNDArray xOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, x);
                 OpaqueNDArray scalarOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, scalar);
                 OpaqueNDArray zOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, z);
                 OpaqueNDArray dimensionsOpaque =
                         dimensions == null || dimensions.data() == null
                                 ? null
                                 : OpaqueNDArray.fromINDArrayUncached(
                                         runtime, dimensions.castTo(DataType.LONG))) {
                if (dimensionsOpaque == null) {
                    switch (op.getOpType()) {
                        case SCALAR:
                            nativeOps.execScalar(
                                    null, op.opNum(), xOpaque, zOpaque,
                                    scalarOpaque, extraArguments);
                            break;
                        case SCALAR_BOOL:
                            nativeOps.execScalarBool(
                                    null, op.opNum(), xOpaque, zOpaque,
                                    scalarOpaque, extraArguments);
                            break;
                        default:
                            throw new UnsupportedOperationException(
                                    "Unknown Vulkan scalar family " + op.getOpType());
                    }
                } else {
                    switch (op.getOpType()) {
                        case SCALAR:
                            nativeOps.execScalarTad(
                                    null, op.opNum(), xOpaque, zOpaque,
                                    scalarOpaque, extraArguments, dimensionsOpaque);
                            break;
                        case SCALAR_BOOL:
                            nativeOps.execScalarBoolTad(
                                    null, op.opNum(), xOpaque, zOpaque,
                                    scalarOpaque, extraArguments, dimensionsOpaque);
                            break;
                        default:
                            throw new UnsupportedOperationException(
                                    "Unknown Vulkan scalar family " + op.getOpType());
                    }
                }
                checkNativeError();
            }
            markDeviceWritten(Collections.singletonList(z));
        } finally {
            if (previousDevice != deviceId) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    private void executeReduction(ReduceOp op, OpContext opContext) {
        INDArray x = getX(op, opContext);
        if (x == null) {
            throw new IllegalArgumentException("Vulkan reduction requires an input array");
        }

        INDArray dimensionArray = op.dimensions();
        long[] dimensions = dimensionArray != null && dimensionArray.data() != null
                ? dimensionArray.toLongVector() : null;
        dimensions = Shape.normalizeAxis(x.rank(), dimensions);
        if (Shape.wholeArrayDimension(dimensions)) {
            dimensions = new long[0];
        }

        int deviceId = affinityManager.getDeviceForArray(x);
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        try {
            if (previousDevice != deviceId) {
                affinityManager.setDeviceForCurrentThread(deviceId);
            }

            long[] resultShape =
                    Shape.reductionShape(x, dimensions, true, op.isKeepDims());
            INDArray z = getZ(op, opContext);
            if (z == null || z == x) {
                z = Nd4j.createUninitialized(op.resultType(), resultShape);
                setZ(z, op, opContext);
            }
            requireSameDevice(z, deviceId, "output");
            op.validateDataTypes(opContext);

            DataType extraType = op.getOpType() == Op.Type.REDUCE_BOOL
                    || op.getOpType() == Op.Type.REDUCE_LONG
                    ? x.dataType() : z.dataType();
            DataBuffer extraBuffer = op.extraArgs() == null || op.extraArgs().length == 0
                    ? null : op.extraArgsDataBuff(extraType);
            Pointer extraArguments =
                    extraBuffer == null ? null : extraBuffer.addressPointer();
            INDArray nativeDimensions = Nd4j.createFromArray(dimensions);
            VulkanRuntime runtime = VulkanRuntime.forNativeOps(nativeOps);

            nativeOps.clearLastError();
            try (OpaqueNDArray xOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, x);
                 OpaqueNDArray yOpaque =
                         OpaqueNDArray.fromINDArrayUncached(
                                 runtime, getY(op, opContext));
                 OpaqueNDArray zOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, z);
                 OpaqueNDArray dimensionsOpaque =
                         OpaqueNDArray.fromINDArrayUncached(
                                 runtime, nativeDimensions)) {
                if (op instanceof Variance) {
                    if (z.isScalar()) {
                        nativeOps.execSummaryStatsScalar(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque,
                                ((Variance) op).isBiasCorrected());
                    } else {
                        nativeOps.execSummaryStatsTad(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque,
                                dimensionsOpaque, ((Variance) op).isBiasCorrected());
                    }
                } else if (yOpaque != null && op.getOpType() == Op.Type.REDUCE3) {
                    if (z.isScalar()) {
                        nativeOps.execReduce3Scalar(
                                null, op.opNum(), xOpaque, extraArguments,
                                yOpaque, zOpaque);
                    } else if (op.isComplexAccumulation()) {
                        nativeOps.execReduce3All(
                                null, op.opNum(), xOpaque, yOpaque, zOpaque,
                                dimensionsOpaque, extraArguments);
                    } else {
                        nativeOps.execReduce3Tad(
                                null, op.opNum(), xOpaque, extraArguments,
                                yOpaque, zOpaque, dimensionsOpaque);
                    }
                } else if (z.isScalar()) {
                    switch (op.getOpType()) {
                        case REDUCE_FLOAT:
                            nativeOps.execReduceFloat(
                                    null, op.opNum(), xOpaque, extraArguments, zOpaque);
                            break;
                        case REDUCE_BOOL:
                            nativeOps.execReduceBool(
                                    null, op.opNum(), xOpaque, extraArguments,
                                    zOpaque, dimensionsOpaque);
                            break;
                        case REDUCE_LONG:
                            nativeOps.execReduceLong(
                                    null, op.opNum(), xOpaque, extraArguments,
                                    zOpaque, dimensionsOpaque);
                            break;
                        case REDUCE_SAME:
                            nativeOps.execReduceSame(
                                    null, op.opNum(), xOpaque, extraArguments, zOpaque);
                            break;
                        default:
                            throw new UnsupportedOperationException(
                                    "Unknown Vulkan reduction family " + op.getOpType());
                    }
                } else {
                    switch (op.getOpType()) {
                        case REDUCE_FLOAT:
                            nativeOps.execReduceFloat2(
                                    null, op.opNum(), xOpaque, extraArguments,
                                    zOpaque, dimensionsOpaque);
                            break;
                        case REDUCE_BOOL:
                            nativeOps.execReduceBool2(
                                    null, op.opNum(), xOpaque, extraArguments,
                                    zOpaque, dimensionsOpaque);
                            break;
                        case REDUCE_LONG:
                            nativeOps.execReduceLong2(
                                    null, op.opNum(), xOpaque, extraArguments,
                                    zOpaque, dimensionsOpaque);
                            break;
                        case REDUCE_SAME:
                            nativeOps.execReduceSame2(
                                    null, op.opNum(), xOpaque, extraArguments,
                                    zOpaque, dimensionsOpaque);
                            break;
                        default:
                            throw new UnsupportedOperationException(
                                    "Unknown Vulkan reduction family " + op.getOpType());
                    }
                }
                checkNativeError();
            }
            markDeviceWritten(Collections.singletonList(z));
        } finally {
            if (previousDevice != deviceId) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    private void executeTransform(TransformOp op, OpContext opContext) {
        INDArray x = getX(op, opContext);
        INDArray y = getY(op, opContext);
        INDArray z = getZ(op, opContext);
        if (x == null || z == null) {
            throw new IllegalArgumentException(
                    "Vulkan transform execution requires explicit input and output arrays");
        }

        int deviceId = affinityManager.getDeviceForArray(x);
        requireSameDevice(y, deviceId, "second input");
        requireSameDevice(z, deviceId, "output");
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        try {
            if (previousDevice != deviceId) {
                affinityManager.setDeviceForCurrentThread(deviceId);
            }
            executeTransformOnCurrentDevice(op, opContext, x, y, z);
        } finally {
            if (previousDevice != deviceId) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    private void executeTransformOnCurrentDevice(
            TransformOp op, OpContext opContext, INDArray x, INDArray y, INDArray z) {
        op.validateDataTypes(opContext, nativeOps.isExperimentalEnabled());
        DataType extraType = op.getOpType() == Op.Type.TRANSFORM_BOOL
                || op.getOpType() == Op.Type.PAIRWISE_BOOL
                ? x.dataType() : z.dataType();
        DataBuffer extraBuffer = op.extraArgs() == null || op.extraArgs().length == 0
                ? null : op.extraArgsDataBuff(extraType);
        Pointer extraArguments = extraBuffer == null ? null : extraBuffer.addressPointer();
        VulkanRuntime runtime = VulkanRuntime.forNativeOps(nativeOps);

        nativeOps.clearLastError();
        try (OpaqueNDArray xOpaque = OpaqueNDArray.fromINDArrayUncached(runtime, x);
             OpaqueNDArray yOpaque = OpaqueNDArray.fromINDArrayUncached(runtime, y);
             OpaqueNDArray zOpaque = OpaqueNDArray.fromINDArrayUncached(runtime, z)) {
            if (yOpaque != null) {
                switch (op.getOpType()) {
                    case TRANSFORM_BOOL:
                    case PAIRWISE_BOOL:
                        nativeOps.execPairwiseTransformBool(
                                null, op.opNum(), xOpaque, yOpaque, extraArguments, zOpaque);
                        break;
                    case TRANSFORM_ANY:
                    case TRANSFORM_FLOAT:
                    case TRANSFORM_SAME:
                    case TRANSFORM_STRICT:
                        nativeOps.execPairwiseTransform(
                                null, op.opNum(), xOpaque, yOpaque, zOpaque, extraArguments);
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                "Unknown Vulkan pairwise transform type " + op.getOpType());
                }
            } else {
                switch (op.getOpType()) {
                    case TRANSFORM_ANY:
                        nativeOps.execTransformAny(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque);
                        break;
                    case TRANSFORM_FLOAT:
                        nativeOps.execTransformFloat(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque);
                        break;
                    case TRANSFORM_BOOL:
                        nativeOps.execTransformBool(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque);
                        break;
                    case TRANSFORM_SAME:
                        nativeOps.execTransformSame(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque);
                        break;
                    case TRANSFORM_STRICT:
                        nativeOps.execTransformStrict(
                                null, op.opNum(), xOpaque, extraArguments, zOpaque);
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                "Unknown Vulkan unary transform type " + op.getOpType());
                }
            }
            checkNativeError();
        }
        markDeviceWritten(Collections.singletonList(z));
    }

    private void requireSameDevice(INDArray array, int expectedDevice, String role) {
        if (array == null || array.isEmpty() || array.data() == null) {
            return;
        }
        int actualDevice = affinityManager.getDeviceForArray(array);
        if (actualDevice != expectedDevice) {
            throw new IllegalArgumentException(
                    "Vulkan operation " + role + " belongs to device " + actualDevice
                            + ", expected device " + expectedDevice);
        }
    }

    private int firstArrayDevice(List<INDArray> arrays) {
        if (arrays == null) {
            return -1;
        }
        for (INDArray array : arrays) {
            if (array != null && !array.isEmpty() && array.data() != null) {
                return affinityManager.getDeviceForArray(array);
            }
        }
        return -1;
    }

    private int customOpTargetDevice(CustomOp op) {
        int targetDevice = firstArrayDevice(op.outputArguments());
        return targetDevice >= 0 ? targetDevice : firstArrayDevice(op.inputArguments());
    }

    private void requireArraysOnDevice(
            List<INDArray> arrays, int expectedDevice, String role) {
        if (arrays == null) {
            return;
        }
        for (int index = 0; index < arrays.size(); index++) {
            requireSameDevice(arrays.get(index), expectedDevice, role + "[" + index + "]");
        }
    }

    private void requireCustomOpArraysOnDevice(CustomOp op, int expectedDevice) {
        requireArraysOnDevice(op.inputArguments(), expectedDevice, "input");
        requireArraysOnDevice(op.outputArguments(), expectedDevice, "output");
    }

    @Override
    public INDArray exec(@NonNull RandomOp op, @NonNull Random rng) {
        return executeRandom(op, null, rng);
    }

    private INDArray executeRandom(@NonNull RandomOp op, OpContext opContext,
                                   @NonNull Random rng) {
        INDArray x = getX(op, opContext);
        INDArray y = getY(op, opContext);
        INDArray z = getZ(op, opContext);
        if (z == null) {
            throw new IllegalArgumentException(
                    "Vulkan random execution requires an explicit output array");
        }
        if (op instanceof BaseRandomOp
                && ((BaseRandomOp) op).isTripleArgRngOp()
                && x == null && y == null) {
            // CUDA uses the output as both logical inputs for triple-argument
            // random kernels such as GaussianDistribution.
            x = z;
            y = z;
        }
        if (rng.getStatePointer() == null || rng.getStatePointer().isNull()) {
            throw new IllegalStateException(
                    "Vulkan random execution requires a NativeRandom state pointer");
        }

        int targetDevice = affinityManager.getDeviceForArray(z);
        requireSameDevice(x, targetDevice, "first random input");
        requireSameDevice(y, targetDevice, "second random input");
        int previousDevice = affinityManager.getDeviceForCurrentThread();

        VulkanRuntime runtime = VulkanRuntime.forNativeOps(nativeOps);

        try {
            if (previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(targetDevice);
            }

            DataBuffer extraBuffer = op.extraArgsDataBuff(z.dataType());
            Pointer extraArguments =
                    extraBuffer == null ? null : extraBuffer.addressPointer();
            nativeOps.clearLastError();
            try (OpaqueNDArray xOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, x);
                 OpaqueNDArray yOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, y);
                 OpaqueNDArray zOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, z)) {
                if (x != null && y != null) {
                    nativeOps.execRandom3(
                            null, op.opNum(), rng.getStatePointer(),
                            xOpaque, yOpaque, zOpaque, extraArguments);
                } else if (x != null) {
                    nativeOps.execRandom2(
                            null, op.opNum(), rng.getStatePointer(),
                            xOpaque, zOpaque, extraArguments);
                } else {
                    nativeOps.execRandom(
                            null, op.opNum(), rng.getStatePointer(),
                            zOpaque, extraArguments);
                }
                checkNativeError();
            }
            markDeviceWritten(Collections.singletonList(z));
            return z;
        } finally {
            if (previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    @Override
    public INDArray[] exec(@NonNull CustomOp op) {
        return executeCustom(op, null);
    }

    private INDArray[] executeCustom(CustomOp op, Random random) {
        int targetDevice = customOpTargetDevice(op);
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        if (targetDevice >= 0) {
            requireCustomOpArraysOnDevice(op, targetDevice);
        }

        String allocationContext = op.opName();
        try {
            if (targetDevice >= 0 && previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(targetDevice);
            }
            if (allocationContext != null) {
                nativeOps.setAllocationContext(allocationContext);
            }

            try (VulkanOpContext context =
                         new VulkanOpContext(VulkanRuntime.forNativeOps(nativeOps))) {
                op.setupOpContextFromCustomOp(context);
                boolean shapeOverride = initializeOutputs(op, context);
                if (shapeOverride) {
                    context.purgeForReuse();
                    op.setupOpContextFromCustomOp(context);
                    context.shapeFunctionOverride(true);
                }
                context.markInplace(op.isInplaceCall());
                if (random != null) {
                    context.setRngStates(random.rootState(), random.nodeState());
                }

                op.assertValidForExecution();
                INDArray[] outputs = exec(op, context);
                if (random != null) {
                    Pair<Long, Long> states = context.getRngStates();
                    random.setStates(states.getFirst(), states.getSecond());
                }
                return outputs;
            }
        } finally {
            try {
                nativeOps.clearAllocationContext();
            } finally {
                if (targetDevice >= 0 && previousDevice != targetDevice) {
                    affinityManager.setDeviceForCurrentThread(previousDevice);
                }
            }
        }
    }

    @Override
    public INDArray[] exec(@NonNull CustomOp op, @NonNull OpContext context) {
        VulkanOpContext vulkanContext = requireVulkanContext(context);
        int targetDevice = vulkanContext.targetDevice();
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        requireArraysOnDevice(vulkanContext.getInputArrays(), targetDevice, "input");
        requireArraysOnDevice(vulkanContext.getOutputArrays(), targetDevice, "output");

        try {
            if (previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(targetDevice);
            }

            long descriptorHash = op.opHash();
            getCustomOperations();
            CustomOpDescriptor descriptor = customOpsByHash.get(descriptorHash);
            if (descriptor == null) {
                throw new UnsupportedOperationException(
                        "No native descriptor is registered for hash " + descriptorHash
                                + " (operation " + op.opName() + ")");
            }

            prepareInputs(vulkanContext);
            prepareOutputs(vulkanContext, op.requiresZeroedOutput());

            nativeOps.clearLastError();
            int status = nativeOps.execCustomOp2(null, descriptorHash,
                    vulkanContext.contextPointer());
            checkNativeError();
            if (status != 0) {
                throw new IllegalStateException(
                        "Vulkan execution failed for descriptor hash "
                                + descriptorHash + " with status " + status);
            }

            if (op.isInplaceCall()) {
                markDeviceWritten(vulkanContext.getInputArrays());
            } else {
                markDeviceWritten(vulkanContext.getOutputArrays());
            }
            return vulkanContext.getOutputArrays().toArray(new INDArray[0]);
        } finally {
            if (previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    private VulkanOpContext requireVulkanContext(OpContext context) {
        if (!(context instanceof VulkanOpContext)) {
            throw new IllegalArgumentException(
                    "Vulkan execution requires VulkanOpContext, got "
                            + context.getClass().getName());
        }
        VulkanOpContext vulkanContext = (VulkanOpContext) context;
        if (vulkanContext.nativeOpsAuthority() != nativeOps
                || vulkanContext.contextPointer().backendOwner().nativeOps() != nativeOps) {
            throw new IllegalArgumentException(
                    "VulkanOpContext belongs to a different native backend instance");
        }
        return vulkanContext;
    }

    private void prepareInputs(VulkanOpContext context) {
        if (context.getInputArrays() == null) {
            return;
        }
        for (INDArray input : context.getInputArrays()) {
            VulkanDataBuffer buffer = requireVulkanBuffer(input, "input");
            if (buffer != null) {
                buffer.syncToSpecial();
            }
        }
    }

    private void prepareOutputs(VulkanOpContext context, boolean zero) {
        if (context.getOutputArrays() == null) {
            return;
        }
        for (INDArray output : context.getOutputArrays()) {
            VulkanDataBuffer buffer = requireVulkanBuffer(output, "output");
            if (buffer == null) {
                continue;
            }
            OpaqueDataBuffer opaque = buffer.opaqueBuffer();
            nativeOps.dbAllocateSpecialBuffer(opaque);
            if (output.length() > 0) {
                Pointer special = opaque.specialBuffer();
                if (special == null || special.isNull()) {
                    throw new IllegalStateException(
                            "Vulkan output has no device allocation");
                }
                if (zero) {
                    nativeOps.clearLastError();
                    int initialized = nativeOps.memsetSync(
                            special, 0,
                            Math.multiplyExact(buffer.length(), buffer.getElementSize()),
                            0, null);
                    checkNativeError();
                    if (initialized != 1) {
                        throw new IllegalStateException(
                                "Vulkan output initialization failed without a native error");
                    }
                }
            }
        }
    }

    private VulkanDataBuffer requireVulkanBuffer(INDArray array, String role) {
        if (array == null || array.isEmpty()) {
            return null;
        }
        if (!(array.data() instanceof VulkanDataBuffer)) {
            throw new IllegalArgumentException(
                    "Vulkan " + role + " array uses "
                            + array.data().getClass().getName());
        }
        VulkanDataBuffer buffer = (VulkanDataBuffer) array.data();
        if (buffer.opaqueBuffer().backendOwner().nativeOps() != nativeOps) {
            throw new IllegalArgumentException(
                    "Vulkan " + role + " buffer belongs to a different backend");
        }
        return buffer;
    }

    private void markDeviceWritten(List<INDArray> arrays) {
        if (arrays == null) {
            return;
        }
        for (INDArray array : arrays) {
            VulkanDataBuffer buffer = requireVulkanBuffer(array, "result");
            if (buffer != null) {
                buffer.markDeviceDirty();
            }
        }
    }

    private boolean initializeOutputs(CustomOp op, VulkanOpContext context) {
        if (op.numOutputArguments() != 0 || op.isInplaceCall()) {
            return false;
        }

        List<DataBuffer> shapes = calculateOutputShape(op, context);
        if (shapes.isEmpty()) {
            throw new IllegalStateException(
                    "Descriptor " + op.opHash() + " returned no output shapes");
        }

        List<INDArray> inputs = op.inputArguments();
        for (DataBuffer shapeBuffer : shapes) {
            long[] shapeInfo = shapeBuffer.asLong();
            int inputIndex = ArrayOptionsHelper.getCopyOffsetInputIndex(shapeInfo);
            INDArray output;
            if (inputIndex >= 0 && inputs != null && inputIndex < inputs.size()) {
                INDArray input = inputs.get(inputIndex);
                output = new VulkanNDArray(
                        input.data(),
                        Shape.shape(shapeInfo),
                        Shape.stride(shapeInfo),
                        input.offset(),
                        Shape.elementWiseStride(shapeInfo),
                        Shape.order(shapeInfo),
                        Shape.dataType(shapeInfo),
                        true);
            } else {
                output = createFromDescriptor(shapeBuffer);
            }
            op.addOutputArgument(output);
        }
        return true;
    }

    @Override
    public void commit() {
        OpaqueLaunchContext launchContext = nativeOps.defaultLaunchContext();
        if (launchContext == null || launchContext.isNull()) {
            throw new IllegalStateException("Vulkan backend returned no launch context");
        }
        Pointer executionStream = nativeOps.lcExecutionStream(launchContext);
        if (executionStream == null || executionStream.isNull()) {
            throw new IllegalStateException("Vulkan backend returned no execution stream");
        }

        nativeOps.clearLastError();
        int status = nativeOps.streamSynchronize(executionStream);
        checkNativeError();
        if (status != 1) {
            throw new IllegalStateException(
                    "Vulkan execution-stream synchronization failed with status " + status);
        }
    }

    @Override
    public ExecutionerType type() {
        return ExecutionerType.VULKAN;
    }

    @Override
    public TADManager getTADManager() {
        return tadManager;
    }

    @Override
    public OpContext buildContext() {
        return new VulkanOpContext(VulkanRuntime.forNativeOps(nativeOps));
    }

    @Override
    public synchronized Map<String, CustomOpDescriptor> getCustomOperations() {
        if (customOps != null) {
            return customOps;
        }

        String serialized = nativeOps.getAllCustomOps();
        if (serialized == null || serialized.isEmpty()) {
            log.warn("No native custom operations are available");
            customOpsByHash = Collections.emptyMap();
            customOps = Collections.emptyMap();
            return customOps;
        }

        Map<String, CustomOpDescriptor> descriptors = new HashMap<>();
        Map<Long, CustomOpDescriptor> descriptorsByHash = new HashMap<>();
        Map<Long, String> descriptorNamesByHash = new HashMap<>();
        for (String entry : serialized.split(";")) {
            if (entry == null || entry.isEmpty()) {
                continue;
            }

            String[] fields = entry.split(":");
            if (fields.length < 7) {
                throw new IllegalStateException("Malformed native custom-op descriptor: " + entry);
            }

            String opName = fields[0];
            CustomOpDescriptor descriptor = CustomOpDescriptor.builder()
                    .hash(Long.parseLong(fields[1]))
                    .numInputs(Integer.parseInt(fields[2]))
                    .numOutputs(Integer.parseInt(fields[3]))
                    .allowsInplace(Integer.parseInt(fields[4]) == 1)
                    .numTArgs(Integer.parseInt(fields[5]))
                    .numIArgs(Integer.parseInt(fields[6]))
                    .build();
            CustomOpDescriptor previous = descriptorsByHash.putIfAbsent(
                    descriptor.getHash(), descriptor);
            String previousName = descriptorNamesByHash.putIfAbsent(
                    descriptor.getHash(), opName);
            if (previous != null && !sameDescriptorSignature(previous, descriptor)) {
                throw new IllegalStateException(
                        "Native custom-op hash collision " + descriptor.getHash()
                                + " between " + previousName + " and " + opName);
            }
            descriptors.put(opName, descriptor);
        }

        customOpsByHash = Collections.unmodifiableMap(descriptorsByHash);
        customOps = Collections.unmodifiableMap(descriptors);
        return customOps;
    }

    private static boolean sameDescriptorSignature(
            CustomOpDescriptor left, CustomOpDescriptor right) {
        return left.getNumInputs() == right.getNumInputs()
                && left.getNumOutputs() == right.getNumOutputs()
                && left.isAllowsInplace() == right.isAllowsInplace()
                && left.getNumTArgs() == right.getNumTArgs()
                && left.getNumIArgs() == right.getNumIArgs();
    }

    @Override
    public List<DataBuffer> calculateOutputShape(@NonNull CustomOp op) {
        int targetDevice = customOpTargetDevice(op);
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        if (targetDevice >= 0) {
            requireCustomOpArraysOnDevice(op, targetDevice);
        }

        String allocationContext = op.opName();
        try {
            if (targetDevice >= 0 && previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(targetDevice);
            }
            if (allocationContext != null) {
                nativeOps.setAllocationContext(allocationContext);
            }

            try (VulkanOpContext context =
                         new VulkanOpContext(VulkanRuntime.forNativeOps(nativeOps))) {
                op.setupOpContextFromCustomOp(context);
                return calculateOutputShape(op, context);
            }
        } finally {
            try {
                nativeOps.clearAllocationContext();
            } finally {
                if (targetDevice >= 0 && previousDevice != targetDevice) {
                    affinityManager.setDeviceForCurrentThread(previousDevice);
                }
            }
        }
    }

    @Override
    public List<DataBuffer> calculateOutputShape(
            @NonNull CustomOp op, @NonNull OpContext opContext) {
        VulkanOpContext context = requireVulkanContext(opContext);
        int targetDevice = context.targetDevice();
        int previousDevice = affinityManager.getDeviceForCurrentThread();
        requireArraysOnDevice(context.getInputArrays(), targetDevice, "input");
        requireArraysOnDevice(context.getOutputArrays(), targetDevice, "output");

        try {
            if (previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(targetDevice);
            }
            return calculateOutputShapeOnCurrentDevice(op, context);
        } finally {
            if (previousDevice != targetDevice) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    private List<DataBuffer> calculateOutputShapeOnCurrentDevice(
            CustomOp op, VulkanOpContext context) {
        nativeOps.clearLastError();
        OpaqueShapeList shapeList =
                nativeOps.calculateOutputShapes2(null, op.opHash(), context.contextPointer());
        checkNativeError();
        if (shapeList == null || shapeList.isNull()) {
            throw new IllegalStateException(
                    "Vulkan shape inference returned no shape list for descriptor "
                            + op.opHash());
        }

        List<DataBuffer> result = new ArrayList<>();
        try {
            long size = nativeOps.getShapeListSize(shapeList);
            if (size < 0 || size > Integer.MAX_VALUE) {
                throw new IllegalStateException(
                        "Invalid Vulkan shape-list size " + size + " for descriptor "
                                + op.opHash());
            }
            for (long index = 0; index < size; index++) {
                LongPointer pointer = nativeOps.getShape(shapeList, index);
                if (pointer == null || pointer.isNull()) {
                    throw new IllegalStateException(
                            "Null output shape " + index + " for descriptor " + op.opHash());
                }
                int length = Shape.shapeInfoLength(Math.toIntExact(pointer.get(0)));
                long[] shapeInfo = new long[length];
                pointer.capacity(length).get(shapeInfo, 0, length);
                DataBuffer shapeBuffer = dataBufferFactory.createLong(shapeInfo);
                shapeBuffer.setConstant(true);
                result.add(shapeBuffer);
            }
            return result;
        } finally {
            nativeOps.deleteShapeList(shapeList);
        }
    }

    @Override
    public Properties getEnvironmentInformation() {
        Properties properties = super.getEnvironmentInformation();
        properties.put("backend", "Vulkan");
        properties.put("blas.vendor", "Vulkan device kernels");
        properties.put("devices", nativeOps.getAvailableDevices());
        return properties;
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        if (!(buffer instanceof VulkanDataBuffer)) {
            throw new IllegalArgumentException(
                    "Expected VulkanDataBuffer but received " + buffer.getClass().getName());
        }
        return ((VulkanDataBuffer) buffer).getString(index);
    }

    @Override
    public DataBuffer createShapeInfo(
            long[] shape, long[] stride, long elementWiseStride, char order, DataType dataType, boolean empty) {
        checkNativeError();
        long extras = ArrayOptionsHelper.composeTypicalChecks(
                empty, dataType, false, false, false, false, false);
        return createShapeInfo(shape, stride, elementWiseStride, order, dataType, extras);
    }

    @Override
    public DataBuffer createShapeInfo(
            long[] shape, long[] stride, long elementWiseStride, char order,
            DataType dataType, boolean empty, boolean view) {
        checkNativeError();
        long extras = ArrayOptionsHelper.composeTypicalChecks(
                empty, dataType, false, false, view, false, false);
        return createShapeInfo(shape, stride, elementWiseStride, order, dataType, extras);
    }

    @Override
    public DataBuffer createShapeInfo(
            long[] shape, long[] stride, long elementWiseStride, char order,
            DataType dataType, long extras) {
        OpaqueConstantShapeBuffer shapeBuffer = nativeOps.shapeBufferEx(
                shape.length,
                new LongPointer(shape),
                new LongPointer(stride),
                dataType.toInt(),
                order,
                elementWiseStride,
                extras);
        checkNativeError();
        shapeBuffer.retainReference();

        Pointer primary = nativeOps.getConstantShapeBufferPrimary(shapeBuffer);
        Pointer special = nativeOps.getConstantShapeBufferSpecial(shapeBuffer);
        VulkanDataBuffer result = new VulkanDataBuffer(
                DataType.INT64, primary, special, null, Shape.shapeInfoLength(shape.length));
        result.setConstant(true);
        return result;
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimensions) {
        checkNativeError();
        OpaqueTadPack nativePack = nativeOps.tadOnlyShapeInfo(
                array.shapeInfoDataBuffer().opaqueBuffer(),
                new LongPointer(ArrayUtil.toLongArray(dimensions)),
                dimensions.length).retainReference();
        checkNativeError();

        LongPointer primaryShape = nativeOps.getPrimaryShapeInfo(nativePack).retainReference();
        LongPointer specialShape = nativeOps.getSpecialShapeInfo(nativePack).retainReference();
        LongPointer primaryOffsets = nativeOps.getPrimaryOffsets(nativePack).retainReference();
        LongPointer specialOffsets = nativeOps.getSpecialOffsets(nativePack).retainReference();

        VulkanDataBuffer shapeInfo = new VulkanDataBuffer(
                DataType.INT64,
                primaryShape,
                specialShape,
                null,
                nativeOps.getShapeInfoLength(nativePack));
        VulkanDataBuffer offsets = new VulkanDataBuffer(
                DataType.INT64,
                primaryOffsets,
                specialOffsets,
                null,
                nativeOps.getNumberOfTads(nativePack));
        shapeInfo.setConstant(true);
        offsets.setConstant(true);
        return new TadPack(shapeInfo, offsets);
    }

    @Override
    public INDArray createFromDescriptor(DataBuffer shapeInformation) {
        long[] shapeInfo = shapeInformation.asLong();
        VulkanNDArray array = new VulkanNDArray();
        array.setShapeInfoDataBuffer(shapeInformation);
        DataType dataType = Shape.dataType(shapeInfo);
        long length = Shape.isEmpty(shapeInfo) ? 0L : Shape.length(shapeInfo);
        array.setData(dataBufferFactory.create(dataType, length, false));
        return array;
    }

    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        checkNativeError();
        OpaqueConstantDataBuffer constant = nativeOps.constantBufferLong(
                desiredType.toInt(), new LongPointer(values), values.length);
        constant.retainReference();
        checkNativeError();

        VulkanDataBuffer buffer = new VulkanDataBuffer(
                desiredType,
                nativeOps.getConstantDataBufferPrimary(constant),
                nativeOps.getConstantDataBufferSpecial(constant),
                null,
                values.length);
        buffer.setConstant(true);
        return buffer;
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType) {
        checkNativeError();
        OpaqueConstantDataBuffer constant = nativeOps.constantBufferDouble(
                desiredType.toInt(), new DoublePointer(values), values.length);
        constant.retainReference();
        checkNativeError();

        VulkanDataBuffer buffer = new VulkanDataBuffer(
                desiredType,
                nativeOps.getConstantDataBufferPrimary(constant),
                nativeOps.getConstantDataBufferSpecial(constant),
                null,
                values.length);
        buffer.setConstant(true);
        return buffer;
    }
}
