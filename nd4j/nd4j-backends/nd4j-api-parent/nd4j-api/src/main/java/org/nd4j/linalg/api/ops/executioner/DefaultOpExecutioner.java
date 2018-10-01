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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.profiler.OpProfiler;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each
 * respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
@Slf4j
public class DefaultOpExecutioner implements OpExecutioner {

    private static final String SCOPE_PANIC_MSG = "For more details, see the ND4J User Guide: nd4j.org/userguide#workspaces-panic";

    protected ProfilingMode profilingMode = ProfilingMode.SCOPE_PANIC;
    protected ExecutionMode executionMode = ExecutionMode.JAVA;

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
    public Op exec(Op op) {
        if (op.isPassThrough()) {
            op.exec();
            return op;
        }

        throw new IllegalStateException("Java computation no longer supported");
    }

    @Override
    public INDArray execAndReturn(Op op) {
        if (op instanceof TransformOp) {
            return execAndReturn((TransformOp) op);
        }
        if (op instanceof ScalarOp) {
            return execAndReturn((ScalarOp) op);
        }
        if (op instanceof ReduceOp) {
            return Nd4j.scalar(execAndReturn((ReduceOp) op).getFinalResult());
        }
        if (op instanceof IndexAccumulation) {
            return Nd4j.scalar(execAndReturn((IndexAccumulation) op).getFinalResult());
        }

        throw new IllegalArgumentException("Illegal opType of op: " + op.getClass());
    }

    @Override
    public void iterateOverAllRows(Op op) {
        //column and row vectors should be treated the same
        if (op.x().isVector()) {
            //reset the op in case
            op.setX(op.x());
            if (op.y() != null)
                op.setY(op.y());
            op.setZ(op.z());
            exec(op);
        }
        //execute row wise
        else if (op.x().isMatrix()) {
            INDArray original = op.x();
            INDArray originalZ = op.z();
            INDArray y = op.y();

            for (int i = 0; i < original.rows(); i++) {
                INDArray row = original.getRow(i);
                INDArray zRow = originalZ.getRow(i);
                op.setX(row.dup());
                op.setZ(zRow.dup());
                if (y != null)
                    op.setY(y.getRow(i).dup());
                exec(op);
                zRow.assign(op.z());
            }
        } else {
            INDArray originalX = op.x();
            INDArray originalZ = op.z();
            for (int i = 0; i < originalX.slices(); i++) {
                INDArray slice = originalX.slice(i);
                INDArray zSlice = originalZ.slice(i);
                op.setX(slice);
                op.setZ(zSlice);
                iterateOverAllRows(op);
            }
        }
    }

    @Override
    public void iterateOverAllColumns(Op op) {
        if (op.x().isVector()) {
            exec(op);
        }
        //execute row wise
        else if (op.x().isMatrix() || op.x().isColumnVector()) {
            exec(op, 1);
        } else {
            INDArray originalX = op.x();
            INDArray originalZ = op.z();
            INDArray y = op.y();
            for (int i = 0; i < op.x().slices(); i++) {
                op.setX(originalX.getColumn(i));
                op.setZ(originalZ.getColumn(i));
                if (y != null)
                    op.setY(y.getColumn(i));
                iterateOverAllColumns(op);
            }
        }
    }


    @Override
    public INDArray execAndReturn(TransformOp op) {
        Op result = exec(op);
        TransformOp t = (TransformOp) result;
        return t.z();
    }


    @Override
    public ReduceOp execAndReturn(ReduceOp op) {
        return (ReduceOp) exec(op);
    }

    @Override
    public ReduceOp execAndReturn(Variance op, boolean biasCorrected) {
        return null;
    }

    @Override
    public INDArray execAndReturn(ScalarOp op) {
        return exec(op).z();
    }

    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        return (IndexAccumulation) exec(op);
    }

    @Override
    public INDArray execAndReturn(BroadcastOp op) {
        return exec(op).z();
    }

    /**
     * Execute and return the result from a vector op
     *
     * @param op
     */
    @Override
    public INDArray execAndReturn(ShapeOp op) {
        exec(op);
        return op.z();
    }

    @Override
    public Op exec(Op op, int... dimension) {
        //do op along all dimensions
        if (dimension.length == op.x().rank()) {
            dimension = new int[] {Integer.MAX_VALUE};
        }

        if (op.isPassThrough()) {
            op.exec(dimension);
            return op;
        }

        if (op instanceof ReduceOp || op instanceof IndexAccumulation) {
            //Overloaded exec(ReduceOp,int...) and exec(IndexAccumulation,int...) should always be called instead of this
            throw new IllegalStateException(
                            "exec(Op,int...) should never be invoked for ReduceOp/IndexAccumulation");
        }
        if (op instanceof ScalarOp) {
            //Scalar op along dimension should be same as on the entire NDArray
            throw new IllegalStateException("Java computation no longer supported");
        }
        if (op instanceof TransformOp) {
            throw new UnsupportedOperationException(
                            "Executing transform ops along a dimension should be done via exec special");
        }
        throw new UnsupportedOperationException("Unknown op opType");
    }

    @Override
    public INDArray exec(ReduceOp op, int... dimension) {

        throw new UnsupportedOperationException("Java computation no longer supported");
    }

    @Override
    public INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension) {
        accumulation.setBiasCorrected(biasCorrected);
        return exec(accumulation, dimension);
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        throw new UnsupportedOperationException("Operation should use exec special");

    }

    @Override
    public ExecutionMode executionMode() {
        return executionMode;
    }

    @Override
    public void setExecutionMode(ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }



    @Override
    public INDArray exec(BroadcastOp broadcast, int... dimension) {
        if (dimension.length == broadcast.x().rank()) {
            dimension = new int[] {Integer.MAX_VALUE};
        }

        if (broadcast.isPassThrough()) {
            broadcast.exec(dimension);
            return broadcast.z();
        }

        throw new IllegalStateException("Java computation no longer supported");

    }

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

    /**
     * @param op
     */
    @Override
    public void exec(ShapeOp op) {
        if(!op.isExecSpecial()) {
            throw new IllegalArgumentException("Only special execution supported right now.");
        }

        op.exec();
    }

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
    public INDArray exec(RandomOp op, Random rng) {
        throw new UnsupportedOperationException();
    }


    @Override
    public void setProfilingMode(ProfilingMode mode) {
        profilingMode = mode;
    }

    @Override
    public ProfilingMode getProfilingMode() {
        return profilingMode;
    }

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
            case DISABLED:
            default:
                return 0L;
        }

        return System.nanoTime();
    }

    protected void checkWorkspace(String opName, INDArray array) {
        if (array.isAttached()) {
            val ws = array.data().getParentWorkspace();

            if (ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {

                if (!ws.isScopeActive()) {
                    throw new ND4JIllegalStateException("Op [" + opName + "] X argument uses leaked workspace pointer from workspace ["
                            + ws.getId() + "]\nAll open workspaces: " + allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
                }

                if (ws.getGenerationId() != array.data().getGenerationId())
                    throw new ND4JIllegalStateException("Op [" + opName + "] X argument uses outdated workspace pointer from workspace ["
                            + ws.getId() + "]\nAll open workspaces: " + allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
            }
        }
    }

    protected void checkForWorkspaces(CustomOp op) {
        for (val input: op.inputArguments())
            checkWorkspace(op.opName(), input);

        for (val output: op.outputArguments())
            checkWorkspace(op.opName(), output);
    }

    protected void checkForWorkspaces(Op op) {
        val x = op.x();
        if (x != null)
            checkWorkspace(op.opName(), x);

        val y = op.y();
        if (y != null)
            checkWorkspace(op.opName(), y);

        val z = op.z();
        if (z != null)
            checkWorkspace(op.opName(), z);
    }

    private static List<String> allOpenWorkspaces(){
        List<MemoryWorkspace> l = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        List<String> workspaces = new ArrayList<>(l.size());
        for( MemoryWorkspace ws : l){
            if(ws.isScopeActive()) {
                workspaces.add(ws.getId());
            }
        }
        return workspaces;
    }

    public long profilingHookIn(Op op) {
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
                checkForWorkspaces(op);
                return 0L;
            case DISABLED:
            default:
                return 0L;
        }

        return System.nanoTime();
    }

    public long profilingHookIn(CustomOp op) {
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
                checkForWorkspaces(op);
                return 0L;
            case DISABLED:
            default:
                return 0L;
        }

        return System.nanoTime();
    }

    public void profilingHookOut(Op op, long timeStart) {
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
                OpExecutionerUtil.checkForNaN(op);
                break;
            case INF_PANIC:
                OpExecutionerUtil.checkForInf(op);
                break;
            case ANY_PANIC:
                OpExecutionerUtil.checkForNaN(op);
                OpExecutionerUtil.checkForInf(op);
                break;
            case DISABLED:
            default:
                break;
        }

        if (Nd4j.getExecutioner().isVerbose()) {
            if (op.z() != null)
                log.info("Z shapeInfo: {}; Z values: {}", op.z().shapeInfoJava(), firstX(op.z(), 10));

            System.out.println();
        }
    }


    public void profilingHookOut(CustomOp op, long timeStart) {
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
                OpExecutionerUtil.checkForNaN(op);
                break;
            case INF_PANIC:
                OpExecutionerUtil.checkForInf(op);
                break;
            case ANY_PANIC:
                OpExecutionerUtil.checkForNaN(op);
                OpExecutionerUtil.checkForInf(op);
                break;
            case DISABLED:
            default:
                break;
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

        if (op.x() != null && !Shape.isEmpty(op.x().shapeInfoJava())
                && op.x().data().dataType() != expectedType
                && op.x().data().dataType() != DataType.COMPRESSED)
            throw new ND4JIllegalStateException("op.X dataType is [" + op.x().data().dataType()
                            + "] instead of expected [" + expectedType + "]");

        if (op.z() != null && !Shape.isEmpty(op.z().shapeInfoJava())
                        && op.z().data().dataType() != expectedType
                        && op.z().data().dataType() != DataType.COMPRESSED)
            throw new ND4JIllegalStateException("op.Z dataType is [" + op.z().data().dataType()
                            + "] instead of expected [" + expectedType + "]");

        if (op.y() != null && !Shape.isEmpty(op.y().shapeInfoJava())
                && op.y().data().dataType() != expectedType)
            throw new ND4JIllegalStateException("op.Y dataType is [" + op.y().data().dataType()
                            + "] instead of expected [" + expectedType + "]");


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

    public static void validateDataType(DataType expectedType, INDArray... operands) {
        if (operands == null || operands.length == 0)
            return;

        int cnt = 0;
        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            if (operand.data().dataType() != expectedType)
                throw new ND4JIllegalStateException("INDArray [" + cnt++ + "] dataType is [" + operand.data().dataType()
                                + "] instead of expected [" + expectedType + "]");
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
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray bitmapEncode(INDArray indArray, double threshold) {
        DataBuffer buffer = Nd4j.getDataBufferFactory().createInt(indArray.length() / 16 + 5);

        INDArray ret = Nd4j.createArrayFromShapeBuffer(buffer, indArray.shapeInfoDataBuffer());

        bitmapEncode(indArray, ret, threshold);

        return ret;
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public Map<String, CustomOpDescriptor> getCustomOperations() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void exec(CustomOp op) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<long[]> calculateOutputShape(CustomOp op) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray[] allocateOutputArrays(CustomOp op){
        List<long[]> shapes = calculateOutputShape(op);
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
}
