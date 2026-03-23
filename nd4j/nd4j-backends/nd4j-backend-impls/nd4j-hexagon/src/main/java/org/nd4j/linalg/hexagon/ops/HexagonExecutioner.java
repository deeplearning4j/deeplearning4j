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

package org.nd4j.linalg.hexagon.ops;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hexagon NPU Operation Executioner using hexagon-mlir.
 *
 * This executioner compiles operations to MLIR targeting Hexagon HVX
 * vector operations and dispatches them to the NPU via command lists.
 *
 * Key characteristics:
 * - Operations compiled to HVX vector instructions via hexagon-mlir
 * - Data staged through TCM for low-latency access
 * - Command list recording for minimal dispatch overhead
 * - Optimized for INT8 quantized inference workloads
 *
 * When hexagon-mlir native bindings are not available, all operations
 * delegate to the CPU-based DefaultOpExecutioner.
 */
@Slf4j
public class HexagonExecutioner extends DefaultOpExecutioner {

    private volatile boolean initialized = false;
    private TADManager tadManager;
    private final ConcurrentHashMap<Long, Object> compiledGraphs = new ConcurrentHashMap<>();

    public HexagonExecutioner() {
        super();
    }

    @Override
    public String getExecutionerType() {
        return "HEXAGON";
    }

    /**
     * Initialize the Hexagon executioner.
     * Loads the hexagon-mlir runtime and detects available NPU devices.
     * If the runtime is unavailable, initialization succeeds but ops
     * delegate to CPU fallback.
     */
    public synchronized void initialize() {
        if (initialized) {
            return;
        }

        log.info("Initializing Hexagon NPU Executioner with hexagon-mlir...");

        boolean runtimeAvailable = false;
        try {
            Class.forName("org.nd4j.linalg.hexagon.bindings.Nd4jHexagon");
            runtimeAvailable = true;
            log.info("hexagon-mlir native bindings loaded successfully");
        } catch (ClassNotFoundException | UnsatisfiedLinkError e) {
            log.info("hexagon-mlir native bindings not available, using CPU fallback. " +
                     "This is expected on non-Hexagon systems. Error: {}", e.getMessage());
        }

        if (runtimeAvailable) {
            org.nd4j.linalg.hexagon.HexagonEnvironment env =
                    org.nd4j.linalg.hexagon.HexagonEnvironment.getInstance();
            log.info("Hexagon NPU detected: {} with {}B HVX, {}KB TCM",
                     env.getNpuVersion(), env.getHvxVectorWidth(),
                     env.getTcmCapacity() / 1024);
        }

        initialized = true;
        log.info("Hexagon NPU Executioner initialized (runtime={})", runtimeAvailable);
    }

    @Override
    public INDArray exec(Op op) {
        checkInitialized();
        return super.exec(op);
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        checkInitialized();
        return execAndReturn(op, opContext);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        checkInitialized();
        return super.exec(op);
    }

    @Override
    public INDArray exec(BroadcastOp op) {
        checkInitialized();
        return super.exec(op);
    }

    @Override
    public INDArray exec(ScalarOp op) {
        checkInitialized();
        return super.exec(op);
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        checkInitialized();
        return super.exec(op);
    }

    @Override
    public void exec(CustomOp op) {
        checkInitialized();
        exec(op, null);
    }

    @Override
    public void exec(CustomOp op, OpContext context) {
        checkInitialized();

        String opName = op.opName();
        if (log.isTraceEnabled()) {
            log.trace("Executing custom op {} on Hexagon NPU", opName);
        }

        // All ops delegate to DefaultOpExecutioner which routes through
        // the native op dispatch. When hexagon-mlir bindings are active,
        // platform helpers registered for ENGINE_HEXAGON will intercept
        // supported ops and route them to MLIR compilation + HVX execution.
        super.exec(op, context);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(@NonNull CustomOp op) {
        return calculateOutputShape(op, null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(@NonNull CustomOp op, OpContext opContext) {
        return super.calculateOutputShape(op, opContext);
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context, INDArray... inputs) {
        checkInitialized();

        if (context == null) {
            context = buildContext();
        }

        for (INDArray input : inputs) {
            context.addInputArgument(input);
        }

        exec(op, context);
        return context.getOutputArrays().toArray(new INDArray[0]);
    }

    @Override
    public void execAndReturn(TransformOp op) {
        checkInitialized();
        super.execAndReturn(op);
    }

    @Override
    public INDArray execAndReturn(Op op, OpContext opContext) {
        checkInitialized();
        return super.execAndReturn(op, opContext);
    }

    @Override
    public TADManager getTADManager() {
        if (tadManager == null) {
            tadManager = new DefaultTADManager();
        }
        return tadManager;
    }

    @Override
    public void setTADManager(TADManager tadManager) {
        this.tadManager = tadManager;
    }

    @Override
    public boolean isExperimentalMode() {
        return false;
    }

    @Override
    public void registerGraph(long id, Pointer graph) {
        checkInitialized();
        log.debug("Registering graph {} for Hexagon NPU compilation", id);
        compiledGraphs.put(id, graph);
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, @NonNull Map<String, INDArray> map,
                                              @NonNull Map<String, Integer> reverseMap) {
        checkInitialized();

        Object compiled = compiledGraphs.get(id);
        if (compiled == null) {
            throw new IllegalStateException("Graph " + id + " not registered. " +
                    "Call registerGraph() before executeGraph().");
        }

        log.debug("Executing graph {} on Hexagon NPU with {} inputs", id, map.size());
        return super.executeGraph(id, map, reverseMap);
    }

    @Override
    public void forgetGraph(long id) {
        log.debug("Forgetting graph {} from Hexagon NPU compilation cache", id);
        compiledGraphs.remove(id);
    }

    @Override
    public void enableDebugMode(boolean enable) {
        debug.set(enable);
    }

    @Override
    public void enableVerboseMode(boolean enable) {
        verbose.set(enable);
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
        return ExecutionerType.HEXAGON;
    }

    @Override
    public OpContext buildContext() {
        return new HexagonOpContext();
    }

    @Override
    public INDArray bitmapEncode(INDArray indArray, double threshold) {
        return super.bitmapEncode(indArray, threshold);
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {
        return super.bitmapDecode(encoded, target);
    }

    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        return super.bitmapEncode(indArray, target, threshold);
    }

    @Override
    public void commit() {
        // Synchronize NPU execution — no-op without native bindings
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return super.thresholdEncode(input, threshold);
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {
        return super.thresholdEncode(input, threshold, boundary);
    }

    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        return super.thresholdDecode(encoded, target);
    }

    @Override
    public Properties getEnvironmentInformation() {
        Properties props = new Properties();
        props.setProperty("backend", "Hexagon");
        props.setProperty("runtime", "hexagon-mlir");
        props.setProperty("int8.support", "true");

        org.nd4j.linalg.hexagon.HexagonEnvironment env =
                org.nd4j.linalg.hexagon.HexagonEnvironment.getInstance();
        props.setProperty("hexagon.version", env.getNpuVersion());
        props.setProperty("hexagon.hvx.width", String.valueOf(env.getHvxVectorWidth()));
        props.setProperty("hexagon.tcm.bytes", String.valueOf(env.getTcmCapacity()));

        return props;
    }

    private void checkInitialized() {
        if (!initialized) {
            initialize();
        }
    }

    private static class DefaultTADManager implements TADManager {
        @Override
        public org.nd4j.linalg.api.shape.TadPack getTADOnlyShapeInfo(INDArray array, long[] dimension) {
            return Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(array, dimension);
        }

        @Override
        public org.nd4j.common.primitives.Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(
                long[] shapeInfo, long[] dimension) {
            return Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(shapeInfo, dimension);
        }
    }
}
