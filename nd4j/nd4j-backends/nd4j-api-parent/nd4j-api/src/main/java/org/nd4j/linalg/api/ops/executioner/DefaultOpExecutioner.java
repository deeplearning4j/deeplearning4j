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

package org.nd4j.linalg.api.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.compression.DecodeBitmap;
import org.nd4j.linalg.api.ops.compression.DecodeThreshold;
import org.nd4j.linalg.api.ops.compression.EncodeBitmap;
import org.nd4j.linalg.api.ops.compression.EncodeThreshold;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Optional;
import org.nd4j.linalg.profiler.OpProfiler;
import org.nd4j.linalg.profiler.ProfilerConfig;
import org.nd4j.common.util.ArrayUtil;

import java.util.*;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each
 * respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class DefaultOpExecutioner implements OpExecutioner {

    private static final String SCOPE_PANIC_MSG = "For more details, see the ND4J User Guide: https://deeplearning4j.konduit.ai/nd4j/overview#workspaces-scope-panic";

    protected ProfilingMode profilingMode = ProfilingMode.SCOPE_PANIC;

    protected AtomicBoolean verbose = new AtomicBoolean(false);
    protected AtomicBoolean debug = new AtomicBoolean(false);

    public DefaultOpExecutioner() {}

    protected void checkForCompression(Op op) {
        // check for INT datatype arrays
        interceptIntDataType(op);

        if (op.x() != null && op.x().isCompressed())
            Nd4j.getCompressor().decompressi(op.x());

        if (op.y() != null && op.y().isCompressed())
            Nd4j.getCompressor().decompressi(op.y());

        if (op.z() != null && op.z().isCompressed())
            Nd4j.getCompressor().decompressi(op.z());
    }

    @Override
    public String getLastOp() {
        return "UNKNOWN";
    }

    /**
     * This method checks if any Op operand has data opType of INT, and throws exception if any.
     *
     * @param op
     */
    protected void interceptIntDataType(Op op) {
        // FIXME: Remove this method, after we'll add support for <int> dtype operations
/*
        if (op.x() != null && op.x().data().dataType() == DataType.INT)
            throw new ND4JIllegalStateException(
                            "Op.X contains INT data. Operations on INT dataType are not supported yet");

        if (op.z() != null && op.z().data().dataType() == DataType.INT)
            throw new ND4JIllegalStateException(
                            "Op.Z contains INT data. Operations on INT dataType are not supported yet");

        if (op.y() != null && op.y().data().dataType() == DataType.INT)
            throw new ND4JIllegalStateException(
                            "Op.Y contains INT data. Operations on INT dataType are not supported yet.");
        */
    }

    @Override
    public abstract INDArray exec(Op op);

    @Override
    public abstract INDArray exec(Op op, OpContext opContext);

    @Override
    public Op execAndReturn(Op op) {
        if (op instanceof TransformOp) {
            return execAndReturn((TransformOp) op);
        }
        if (op instanceof ScalarOp) {
            return execAndReturn((ScalarOp) op);
        }
        if (op instanceof ReduceOp) {
            exec((ReduceOp) op);
            return op;
        }
        if (op instanceof IndexAccumulation) {
            exec((IndexAccumulation) op);
            return op;
        }

        throw new IllegalArgumentException("Illegal opType of op: " + op.getClass());
    }

    @Override
    public TransformOp execAndReturn(TransformOp op) {
        exec(op);
        return op;
    }


    @Override
    public ReduceOp execAndReturn(ReduceOp op) {
        exec(op);
        return op;
    }

    @Override
    public Variance execAndReturn(Variance op) {
        exec(op);
        return op;
    }

    @Override
    public ScalarOp execAndReturn(ScalarOp op) {
        exec(op);
        return op;
    }

    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        exec(op);
        return op;
    }

    @Override
    public BroadcastOp execAndReturn(BroadcastOp op) {
        exec(op);
        return op;
    }

    @Override
    public INDArray[] exec(CustomOp op) {
        return execAndReturn(op).outputArguments().toArray(new INDArray[0]);
    }

    @Override
    public abstract INDArray exec(ReduceOp op);

    @Override
    public abstract INDArray exec(Variance accumulation);

    @Override
    public abstract INDArray exec(IndexAccumulation op);

    @Override
    public abstract INDArray exec(BroadcastOp broadcast);

    @Override
    public void exec(MetaOp op) {
        throw new UnsupportedOperationException("MetaOp execution isn't supported for this OpExecutioner yet");
    }

    @Override
    public void exec(GridOp op) {
        throw new UnsupportedOperationException("GridOp execution isn't supported for this OpExecutioner yet");
    }

    @Override
    public <T extends Aggregate> void exec(Batch<T> batch) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void exec(Aggregate op) {
        throw new UnsupportedOperationException();
    }

    @Override
    public abstract INDArray exec(ScalarOp op);

    @Override
    public void exec(List<Aggregate> batch) {
        throw new UnsupportedOperationException();
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
     * This method executes specific RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    @Override
    public abstract INDArray exec(RandomOp op, Random rng);


    @Deprecated
    @Override
    public void setProfilingMode(ProfilingMode mode) {

        profilingMode = mode;
        ProfilerConfig config = null;
        switch (profilingMode) {
            case ALL:
                config = ProfilerConfig.builder().checkWorkspaces(true).checkElapsedTime(true).stackTrace(true).build();
                break;
            case METHODS:
                config = ProfilerConfig.builder().stackTrace(true).build();
                break;
            case OPERATIONS:
                config = ProfilerConfig.builder().stackTrace(true).checkElapsedTime(true).build();
                break;
            case SCOPE_PANIC:
                config = ProfilerConfig.builder().checkWorkspaces(true).build();
                break;
            case ANY_PANIC:
                config = ProfilerConfig.builder().checkForINF(true).checkForNAN(true).build();
                break;
            case INF_PANIC:
                config = ProfilerConfig.builder().checkForINF(true).build();
                break;
            case NAN_PANIC:
                config = ProfilerConfig.builder().checkForNAN(true).build();
                break;
            default:
                config = ProfilerConfig.builder().build();
                break;
        }
        OpProfiler.getInstance().setConfig(config);
    }

    @Override
    public void setProfilingConfig(ProfilerConfig config) {
        OpProfiler.getInstance().setConfig(config);
    }

    @Deprecated
    @Override
    public ProfilingMode getProfilingMode() {
        return profilingMode;
    }

    protected void checkWorkspace(String opName, INDArray array) {
        if (array.isAttached()) {
            val ws = array.data().getParentWorkspace();

            if (ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {

                if (!ws.isScopeActive()) {
                    throw new ND4JIllegalStateException("Op [" + opName + "] X argument uses leaked workspace pointer from workspace ["
                            + ws.getId() + "]: Workspace the array was defined in is no longer open.\nAll open workspaces: " + allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
                }

                if (ws.getGenerationId() != array.data().getGenerationId())
                    throw new ND4JIllegalStateException("Op [" + opName + "] X argument uses outdated workspace pointer from workspace ["
                            + ws.getId() + "]: Workspace array was defined in has been closed and reopened at least once since array creation. Array WS iteration: " +
                            array.data().getGenerationId() + ". Workspace current iteration: " +
                            ws.getGenerationId() + "\nAll open workspaces: " + allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
            }
        }
    }

    protected void checkForWorkspaces(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();

        for (val input: inArgs)
            checkWorkspace(op.opName(), input);

        for (val output: outArgs)
            checkWorkspace(op.opName(), output);
    }

    protected void checkForWorkspaces(Op op, OpContext oc) {
        val x = oc != null ? oc.getInputArray(0) : op.x();
        if (x != null)
            checkWorkspace(op.opName(), x);

        val y = oc != null && oc.getInputArrays().size() > 1 ? oc.getInputArray(1) : op.y();
        if (y != null)
            checkWorkspace(op.opName(), y);

        val z = oc != null ? oc.getOutputArray(0) : op.z();
        if (z != null)
            checkWorkspace(op.opName(), z);
    }

    public static List<String> allOpenWorkspaces(){
        List<MemoryWorkspace> l = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        List<String> workspaces = new ArrayList<>(l.size());
        for( MemoryWorkspace ws : l){
            if(ws.isScopeActive()) {
                workspaces.add(ws.getId());
            }
        }
        return workspaces;
    }

    @Deprecated
    public long profilingHookIn(Op op, DataBuffer... tadBuffers) {
        switch (profilingMode) {
            case ALL:
                OpProfiler.getInstance().processOpCall(op, tadBuffers);
                break;
            case METHODS:
                break;
            case OPERATIONS:
                OpProfiler.getInstance().processOpCall(op, tadBuffers);
                break;
            case SCOPE_PANIC:
                checkForWorkspaces(op, null);
                return 0L;
            case DISABLED:
            default:
                return 0L;
        }

        return System.nanoTime();
    }

    @Deprecated
    public long profilingHookIn(CustomOp op, OpContext oc) {
        switch (profilingMode) {
            case ALL:
                OpProfiler.getInstance().processOpCall(op);
                break;
            case METHODS:
                break;
            case OPERATIONS:
                OpProfiler.getInstance().processOpCall(op);
                break;
            case SCOPE_PANIC:
                checkForWorkspaces(op, oc);
                return 0L;
            case DISABLED:
            default:
                return 0L;
        }

        return System.nanoTime();
    }

    @Deprecated
    public void profilingHookOut(Op op, OpContext oc, long timeStart) {
        switch (profilingMode) {
            case ALL:
                OpProfiler.getInstance().processStackCall(op, timeStart);
                OpProfiler.getInstance().timeOpCall(op, timeStart);
                break;
            case METHODS:
                OpProfiler.getInstance().processStackCall(op, timeStart);
                break;
            case OPERATIONS:
                OpProfiler.getInstance().timeOpCall(op, timeStart);
                break;
            case NAN_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                break;
            case INF_PANIC:
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case ANY_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case DISABLED:
            default:
                break;
        }

        if (Nd4j.getExecutioner().isVerbose()) {
            if (op.z() != null)
                log.info("Op name: {}; Z shapeInfo: {}; Z values: {}", op.opName(), op.z().shapeInfoJava(), firstX(op.z(), 10));
        }
    }

    @Deprecated
    public void profilingHookOut(CustomOp op, OpContext oc, long timeStart) {
        switch (profilingMode) {
            case ALL:
                OpProfiler.getInstance().processStackCall(op, timeStart);
                OpProfiler.getInstance().timeOpCall(op, timeStart);
                break;
            case METHODS:
                OpProfiler.getInstance().processStackCall(op, timeStart);
                break;
            case OPERATIONS:
                OpProfiler.getInstance().timeOpCall(op, timeStart);
                break;
            case NAN_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                break;
            case INF_PANIC:
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case ANY_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case DISABLED:
            default:
                break;
        }
    }


    public long profilingConfigurableHookIn(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();

        for (val arr: inArgs)
            if (arr.wasClosed())
                throw new IllegalStateException("One of Input arguments was closed before call");

        for (val arr: outArgs)
            if (arr.wasClosed())
                throw new IllegalStateException("One of Output arguments was closed before call");

        if (OpProfiler.getInstance().getConfig() == null)
            return System.nanoTime();

        if (OpProfiler.getInstance().getConfig().isStackTrace() ||
            OpProfiler.getInstance().getConfig().isCheckElapsedTime()) {
            OpProfiler.getInstance().processOpCall(op);
        }

        if (OpProfiler.getInstance().getConfig().isCheckWorkspaces()) {
            checkForWorkspaces(op, oc);
        }

        return System.nanoTime();
    }

    public long profilingConfigurableHookIn(Op op, DataBuffer... tadBuffers) {
        if (op.x() != null)
            if (op.x().wasClosed())
                throw new IllegalStateException("Op.X argument was closed before call");

        if (op.y() != null)
            if (op.y().wasClosed())
                throw new IllegalStateException("Op.Y argument was closed before call");

        if (op.z() != null)
            if (op.z().wasClosed())
                throw new IllegalStateException("Op.Z argument was closed before call");

        if (OpProfiler.getInstance().getConfig() == null)
            return System.nanoTime();

        if (OpProfiler.getInstance().getConfig().isStackTrace() ||
            OpProfiler.getInstance().getConfig().isCheckElapsedTime()) {
            OpProfiler.getInstance().processOpCall(op);
        }

        if (OpProfiler.getInstance().getConfig().isNotOptimalTAD()) {
            OpProfiler.getInstance().processOpCall(op, tadBuffers);
        }
        if (OpProfiler.getInstance().getConfig().isCheckWorkspaces()) {
            checkForWorkspaces(op, null);
        }

        return System.nanoTime();
    }


    public void profilingConfigurableHookOut(Op op, OpContext oc, long timeStart) {
        if (OpProfiler.getInstance().getConfig() == null)
            return;

        if (OpProfiler.getInstance().getConfig().isStackTrace()) {
            OpProfiler.getInstance().processStackCall(op, timeStart);
        }
        if (OpProfiler.getInstance().getConfig().isCheckElapsedTime()) {
            OpProfiler.getInstance().timeOpCall(op, timeStart);
        }
        if (OpProfiler.getInstance().getConfig().isCheckForNAN()) {
            OpExecutionerUtil.checkForNaN(op, oc);
        }
        if (OpProfiler.getInstance().getConfig().isCheckForINF()) {
            OpExecutionerUtil.checkForInf(op, oc);
        }
        if (OpProfiler.getInstance().getConfig().isNativeStatistics()) {
            if (op.z() != null) {
                INDArrayStatistics stat = inspectArray(op.z());
                OpProfiler.getInstance().setStatistics(stat);
                log.info("Op name: {}; Z shapeInfo: {}; Statistics: min:{} max:{} mean:{} stdev:{} pos:{}, neg:{} zero:{} inf:{} nan:{}",
                        op.opName(), op.z().shapeInfoJava(), stat.getMinValue(), stat.getMaxValue(), stat.getMeanValue(),
                        stat.getStdDevValue(), stat.getCountPositive(), stat.getCountNegative(),
                        stat.getCountZero(), stat.getCountInf(), stat.getCountNaN());
            }
        }

        if (Nd4j.getExecutioner().isVerbose()) {
            if (op.z() != null)
                log.info("Op name: {}; Z shapeInfo: {}; Z values: {}", op.opName(), op.z().shapeInfoJava(), firstX(op.z(), 10));
        }
    }

    public void profilingConfigurableHookOut(CustomOp op, OpContext oc, long timeStart) {
        if (OpProfiler.getInstance().getConfig() == null)
            return;

        if (OpProfiler.getInstance().getConfig().isStackTrace()) {
            OpProfiler.getInstance().processStackCall(op, timeStart);
        }
        if (OpProfiler.getInstance().getConfig().isCheckElapsedTime()) {
            OpProfiler.getInstance().timeOpCall(op, timeStart);
        }
        if (OpProfiler.getInstance().getConfig().isCheckForNAN()) {
            OpExecutionerUtil.checkForNaN(op, oc);
        }
        if (OpProfiler.getInstance().getConfig().isCheckForINF()) {
            OpExecutionerUtil.checkForInf(op, oc);
        }
    }

    /**
     * Validate the data types
     * for the given operation
     * @param expectedType
     * @param op
     */
    public static void validateDataType(DataType expectedType, Op op) {
        if (op.x() != null && !Shape.isEmpty(op.x().shapeInfoJava()) && op.x().data().dataType() == DataType.COMPRESSED) {
            Nd4j.getCompressor().decompressi(op.x());
        }

        if (op.y() != null && !Shape.isEmpty(op.y().shapeInfoJava()) && op.y().data().dataType() == DataType.COMPRESSED) {
            Nd4j.getCompressor().decompressi(op.y());
        }

        if (op.z() != null && !Shape.isEmpty(op.z().shapeInfoJava()) && op.z().data().dataType() == DataType.COMPRESSED) {
            Nd4j.getCompressor().decompressi(op.z());
        }

        /*
        if (op.x() != null && !Shape.isEmpty(op.x().shapeInfoJava())
                && op.x().data().dataType() != expectedType
                && op.x().data().dataType() != DataType.COMPRESSED) {
            throw new ND4JIllegalStateException("op.X dataType is [" + op.x().data().dataType()
                    + "] instead of expected [" + expectedType + "] - x.shape = " + Arrays.toString(op.x().shape())
                    + (op.y() != null ? ", y.shape=" + Arrays.toString(op.y().shape()) : "")
                    + ", z.shape=" + Arrays.toString(op.z().shape()) + " - op: " + op.getClass().getName());
        }
        */
/*
        if (op.z() != null && !Shape.isEmpty(op.z().shapeInfoJava())
                        && op.z().data().dataType() != expectedType
                        && op.z().data().dataType() != DataType.COMPRESSED)
            throw new ND4JIllegalStateException("op.Z dataType is [" + op.z().data().dataType()
                            + "] instead of expected [" + expectedType + "]");
        */

        if (op.y() != null && !Shape.isEmpty(op.y().shapeInfoJava())
                && op.y().data().dataType() != expectedType) {
            throw new ND4JIllegalStateException("op.Y dataType is [" + op.y().data().dataType()
                    + "] instead of expected [" + expectedType + "] - x.shape = " + Arrays.toString(op.x().shape())
                    + (op.y() != null ? ", y.shape=" + Arrays.toString(op.y().shape()) : "")
                    + ", z.shape=" + Arrays.toString(op.z().shape()) + " - op: " + op.getClass().getName());

        }


        if (Nd4j.getExecutioner().isVerbose()) {
            log.info("Reporting [{}]", op.opName());
            if (op.x() != null)
                log.info("X shapeInfo: {}; X values: {}", op.x().shapeInfoJava(), firstX(op.x(), 10));

            if (op.y() != null)
                log.info("Y shapeInfo: {}; Y values: {}", op.y().shapeInfoJava(), firstX(op.y(), 10));
        }
    }

    protected static String firstX(INDArray array, int x) {
        val builder = new StringBuilder("[");
        val limit = (int) Math.min(x, array.length());
        for (int e = 0; e < limit; e++) {
            builder.append(array.getDouble(e));

            if (e < limit - 1)
                builder.append(", ");
        }
        builder.append("]");

        return builder.toString();
    }

    public static void validateDataType(DataType expectedType, Object op, INDArray... operands) {
        if (operands == null || operands.length == 0)
            return;

        int cnt = 0;
        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            if (operand.data().dataType() != expectedType) {
                throw new ND4JIllegalStateException("INDArray [" + cnt + "] dataType is [" + operand.data().dataType()
                        + "] instead of expected [" + expectedType + "]" + (op != null ? " op: " + op.getClass().getName() : ""));
            }
            cnt++;
        }
    }

    @Override
    public TADManager getTADManager() {
        throw new UnsupportedOperationException();
    }

    /**
     * This method return set of key/value and key/key/value objects, describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        Properties environment = new Properties();
        environment.put(Nd4jEnvironment.CPU_CORES_KEY, Runtime.getRuntime().availableProcessors());
        environment.put(Nd4jEnvironment.HOST_TOTAL_MEMORY_KEY, Runtime.getRuntime().maxMemory());
        environment.put(Nd4jEnvironment.OS_KEY, System.getProperty("os.name"));
        return environment;
    }

    @Override
    public void printEnvironmentInformation() {
        Properties env = getEnvironmentInformation();
        double memory = ((Long) env.get("memory.available")) / (double) 1024 / 1024 / 1024;
        String fm = String.format("%.1f", memory);
        log.info("Backend used: [{}]; OS: [{}]", env.get("backend"), env.get("os"));
        log.info("Cores: [{}]; Memory: [{}GB];", env.get("cores"), fm);
        log.info("Blas vendor: [{}]", env.get("blas.vendor"));
    }

    @Override
    public void push() {
        // no-op
    }

    @Override
    public void commit() {
        // no-op
    }


    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return thresholdEncode(input, threshold, Integer.MAX_VALUE);
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {
        val result = Nd4j.exec(new EncodeThreshold(input, (float) threshold, boundary))[1];

        return result.getInt(0) > 0 ? result : null;
    }

    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        Nd4j.exec(new DecodeThreshold(encoded, target));
        return target;
    }

    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        val results = Nd4j.exec(new EncodeBitmap(indArray, target, Nd4j.scalar(0), (float) threshold));

        // return number of elements taht were compressed
        return results[2].getInt(0);
    }

    @Override
    public INDArray bitmapEncode(INDArray indArray, double threshold) {
        val array = Nd4j.create(DataType.INT32, indArray.length() / 16 + 5);
        bitmapEncode(indArray, array, threshold);
        return array;
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {
        Nd4j.exec(new DecodeBitmap(encoded, target));
        return target;
    }


    @Override
    public Map<String, CustomOpDescriptor> getCustomOperations() {
        throw new UnsupportedOperationException();
    }

    @Override
    public CustomOp execAndReturn(CustomOp op) {
        exec(op);
        return op;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(CustomOp op) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(CustomOp op, OpContext opContext) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray[] allocateOutputArrays(CustomOp op){
        List<LongShapeDescriptor> shapes = calculateOutputShape(op);
        INDArray[] out = new INDArray[shapes.size()];
        for(int i=0; i<shapes.size(); i++ ){
            out[i] = Nd4j.create(shapes.get(i));
        }
        return out;
    }


    @Override
    public void enableDebugMode(boolean reallyEnable) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void registerGraph(long id, Pointer graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, Map<String, INDArray> map, Map<String, Integer> reverseMap) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void forgetGraph(long id) {
        throw new UnsupportedOperationException("Not yet implemented");
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
        // no-op
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
        // no-op
    }

    @Override
    public boolean isVerbose() {
        return verbose.get();
    }

    @Override
    public boolean isDebug() {
        return debug.get();
    }

    @Override
    public ExecutionerType type() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void scatterUpdate(ScatterUpdate.UpdateOp op, INDArray array, INDArray indices, INDArray updates, int[] axis) {
        throw new UnsupportedOperationException();
    }


    /**
     * Get the information about the op in a String representation, for throwing more useful exceptions (mainly for debugging)
     * @param op
     * @param dimensions    Use optional here for 3 states: null = "not an exec(Op, int... dim) call". empty = "exec(Op, null)".
     *                     Otherwise present = "exec(Op,int[])" call
     * @return
     */
    public String opInfoString(Op op, Optional<int[]> dimensions){
        if(op == null)
            return "<NULL OP>";

        StringBuilder sb = new StringBuilder();
        sb.append("Class: ").append(op.getClass().getName()).append("; opNum: ").append(op.opNum())
                .append("; opName: ").append(op.opName());
        if(op instanceof DifferentialFunction){
            sb.append("; opType: ").append(((DifferentialFunction)op).opType());
        }

        if(dimensions != null){
            sb.append("; dimensions: ");
            if(dimensions.isPresent()){
                sb.append(Arrays.toString(dimensions.get()));
            } else {
                sb.append("<null>");
            }
        }

        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();
        Object[] extraArgs = op.extraArgs();

        sb.append("\n");
        sb.append("x: ").append(arrayInfo(x)).append("; ");
        sb.append("y: ").append(arrayInfo(y)).append("; ");
        sb.append("z: ").append(arrayInfo(z)).append("; ");
        if(x == y && x != null)
            sb.append("(x == y)");
        if(x == z && x != null)
            sb.append("(x == z)");
        if(y == z && y != null)
            sb.append("(y == z)");
        sb.append("\n");
        sb.append("; extraArgs: ").append(Preconditions.formatArray(extraArgs));
        return sb.toString();
    }

    public String arrayInfo(INDArray arr){
        if(arr == null)
            return "<null>";
        if(arr.isEmpty())
            return "(empty NDArray)";

        return arr.shapeInfoToString().replaceAll("\n","");
    }

    @Override
    public boolean isExperimentalMode() {
        return false;
    }

    @Override
    public OpContext buildContext() {
        throw new UnsupportedOperationException("OpContext is available only on native backends");
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        throw new UnsupportedOperationException();
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, int[] dimension) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createConstantBuffer(int[] values, DataType desiredType) {
        return createConstantBuffer(ArrayUtil.toLongArray(values), desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(float[] values, DataType desiredType) {
        return createConstantBuffer(ArrayUtil.toDoubles(values), desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType)  {
        throw new UnsupportedOperationException();
    }

    @Override
    public String runLightBenchmarkSuit(boolean printOut) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String runFullBenchmarkSuit(boolean printOut) {
        throw new UnsupportedOperationException();
    }


    public void setX(INDArray x, Op op, OpContext oc){
        if(oc != null)
            oc.setInputArray(0, x);
        else
            op.setX(x);
    }

    public INDArray getX(Op op, OpContext oc){
        if( oc != null )
            return oc.getInputArray(0);
        return op.x();
    }

    public void setY(INDArray y, Op op, OpContext oc){
        if(oc != null)
            oc.setInputArray(1, y);
        else
            op.setY(y);
    }

    public INDArray getY(Op op, OpContext oc){
        if( oc != null )
            return oc.getInputArray(1);
        return op.y();
    }

    public void setZ(INDArray z, Op op, OpContext oc){
        if(oc != null)
            oc.setOutputArray(0, z);
        else
            op.setZ(z);
    }

    public INDArray getZ(Op op, OpContext oc){
        if( oc != null )
            return oc.getOutputArray(0);
        return op.z();
    }
}
