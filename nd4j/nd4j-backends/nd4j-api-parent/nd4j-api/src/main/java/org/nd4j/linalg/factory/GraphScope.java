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

package org.nd4j.linalg.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.LazyINDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.*;

/**
 * Core trace-compile-replay coordinator for the global graph module.
 *
 * Allows eager Nd4j operations to transparently benefit from the DSP
 * (Dynamic Shape Plan) system by tracing ops into a hidden SameDiff graph,
 * compiling the graph into a plan, and executing it in a single optimized pass.
 *
 * Usage:
 * <pre>
 * try (GraphScope scope = Nd4j.graphScope()) {
 *     scope.begin();
 *     INDArray mm = Nd4j.matmul(a, b);
 *     INDArray out = Nd4j.nn.relu(mm);
 *     scope.end();  // compiles + executes, out now has real values
 * }
 * </pre>
 *
 * @author Adam Gibson
 */
@Slf4j
public class GraphScope implements AutoCloseable {

    private static final ThreadLocal<GraphScope> ACTIVE = new ThreadLocal<>();

    private SameDiff graph;
    private IdentityHashMap<INDArray, String> arrayToVar;
    private List<INDArray> externalInputs;
    private List<LazyINDArray> allLazyOutputs;
    private DynamicShapePlan compiledPlan;
    private DynamicShapePlanExecutor executor;
    private String cachedOpSequenceKey;
    private List<String> opSequence;
    private int varCounter;
    private int opCounter;
    // Track which lazy outputs are consumed by other ops (non-leaf)
    private Set<LazyINDArray> consumedOutputs;

    enum State { IDLE, TRACING, EXECUTED }
    private State state = State.IDLE;

    /**
     * Check if a GraphScope is currently tracing on this thread.
     */
    public static boolean isTracing() {
        GraphScope scope = ACTIVE.get();
        return scope != null && scope.state == State.TRACING;
    }

    /**
     * Get the currently active GraphScope for this thread, or null.
     */
    public static GraphScope active() {
        GraphScope scope = ACTIVE.get();
        if (scope != null && scope.state == State.TRACING) {
            return scope;
        }
        return null;
    }

    /**
     * Begin tracing. All subsequent Nd4j.exec() calls on this thread will be
     * recorded into a hidden SameDiff graph instead of being executed eagerly.
     */
    public void begin() {
        if (ACTIVE.get() != null && ACTIVE.get().state == State.TRACING) {
            throw new IllegalStateException("Nested GraphScope tracing is not supported. " +
                    "A GraphScope is already active on this thread.");
        }
        graph = SameDiff.create();
        arrayToVar = new IdentityHashMap<>();
        externalInputs = new ArrayList<>();
        allLazyOutputs = new ArrayList<>();
        opSequence = new ArrayList<>();
        consumedOutputs = new HashSet<>();
        varCounter = 0;
        opCounter = 0;
        state = State.TRACING;
        ACTIVE.set(this);
        log.debug("GraphScope tracing started");
    }

    /**
     * Record an op into the hidden SameDiff graph.
     *
     * For each input:
     * - If LazyINDArray: look up its SDVariable name -> wire as graph intermediate
     * - If real INDArray: register as placeholder (deduplicated by identity)
     *
     * Creates a DynamicCustomOp in the hidden graph, infers output shapes,
     * and returns LazyINDArray proxies for each output.
     */
    public INDArray[] recordOp(CustomOp op) {
        if (state != State.TRACING) {
            throw new IllegalStateException("GraphScope is not in TRACING state");
        }

        String opName = op.opName();
        String uniqueOpName = opName + "_" + (opCounter++);
        opSequence.add(opName);

        // Map inputs to SDVariables
        List<SDVariable> sdInputs = new ArrayList<>();
        INDArray[] inputArrays = op.inputArguments().toArray(new INDArray[0]);

        for (INDArray input : inputArrays) {
            SDVariable sdVar = resolveInputToSDVariable(input);
            sdInputs.add(sdVar);
        }

        // Create the op in the hidden graph
        SDVariable[] sdInputArray = sdInputs.toArray(new SDVariable[0]);
        DynamicCustomOp graphOp = new DynamicCustomOp(opName, graph, sdInputArray);
        graphOp.setOwnName(uniqueOpName);

        // Copy arguments from the original op
        long[] iArgs = op.iArgs();
        if (iArgs != null && iArgs.length > 0) {
            graphOp.addIArgument(iArgs);
        }
        double[] tArgs = op.tArgs();
        if (tArgs != null && tArgs.length > 0) {
            graphOp.addTArgument(tArgs);
        }
        boolean[] bArgs = op.bArgs();
        if (bArgs != null && bArgs.length > 0) {
            graphOp.addBArgument(bArgs);
        }
        DataType[] dArgs = op.dArgs();
        if (dArgs != null && dArgs.length > 0) {
            graphOp.addDArgument(dArgs);
        }

        // Generate output variables in the graph
        SDVariable[] outputVars = graphOp.outputVariables(uniqueOpName);

        // Infer output shapes using the real input shapes
        List<LongShapeDescriptor> outputShapes = inferOutputShapes(op, inputArrays);

        // Create LazyINDArray proxies for each output
        INDArray[] lazyOutputs = new INDArray[outputVars.length];
        for (int i = 0; i < outputVars.length; i++) {
            long[] shape;
            DataType dtype;
            if (outputShapes != null && i < outputShapes.size()) {
                LongShapeDescriptor desc = outputShapes.get(i);
                shape = desc.getShape();
                dtype = desc.dataType();
            } else {
                // Fallback: use FLOAT and empty shape
                shape = new long[0];
                dtype = DataType.FLOAT;
            }

            LazyINDArray lazy = new LazyINDArray(outputVars[i].name(), shape, dtype);
            allLazyOutputs.add(lazy);
            // Register this lazy array so subsequent ops can reference it
            arrayToVar.put(lazy, outputVars[i].name());
            lazyOutputs[i] = lazy;
        }

        log.debug("Recorded op '{}' with {} inputs, {} outputs",
                uniqueOpName, inputArrays.length, outputVars.length);

        return lazyOutputs;
    }

    /**
     * Resolve an input INDArray to an SDVariable in the hidden graph.
     * LazyINDArrays are looked up by their variable name.
     * Real INDArrays are registered as placeholders (deduplicated by identity).
     */
    private SDVariable resolveInputToSDVariable(INDArray input) {
        // Check if this is a LazyINDArray from a previous traced op
        if (input instanceof LazyINDArray) {
            LazyINDArray lazy = (LazyINDArray) input;
            String varName = lazy.getSdVariableName();
            SDVariable existing = graph.getVariable(varName);
            if (existing != null) {
                // Mark this lazy as consumed (not a leaf output)
                consumedOutputs.add(lazy);
                return existing;
            }
            throw new IllegalStateException("LazyINDArray '" + varName
                    + "' not found in graph. Was it created by a different GraphScope?");
        }

        // Check if we've already registered this exact array (by identity)
        String existingName = arrayToVar.get(input);
        if (existingName != null) {
            return graph.getVariable(existingName);
        }

        // Register as a new placeholder
        String placeholderName = "input_" + (varCounter++);
        SDVariable placeholder = graph.placeHolder(placeholderName, input.dataType(), input.shape());
        arrayToVar.put(input, placeholderName);
        externalInputs.add(input);
        return placeholder;
    }

    /**
     * Infer output shapes for the given op using the executioner's shape calculation.
     * Converts from List<DataBuffer> (shape info buffers) to List<LongShapeDescriptor>.
     */
    private List<LongShapeDescriptor> inferOutputShapes(CustomOp op, INDArray[] inputs) {
        try {
            List<org.nd4j.linalg.api.buffer.DataBuffer> shapeBuffers = Nd4j.getExecutioner().calculateOutputShape(op);
            if (shapeBuffers == null || shapeBuffers.isEmpty()) {
                return null;
            }
            List<LongShapeDescriptor> result = new ArrayList<>();
            for (org.nd4j.linalg.api.buffer.DataBuffer shapeBuffer : shapeBuffers) {
                long[] shapeInfo = shapeBuffer.asLong();
                long[] shape = org.nd4j.linalg.api.shape.Shape.shape(shapeInfo);
                DataType dtype = org.nd4j.linalg.api.shape.options.ArrayOptionsHelper.dataType(shapeInfo);
                result.add(LongShapeDescriptor.fromShape(shape, dtype));
            }
            return result;
        } catch (Exception e) {
            log.warn("Failed to infer output shapes for op '{}': {}", op.opName(), e.getMessage());
            return null;
        }
    }

    /**
     * Explicitly mark which arrays are the final outputs.
     * Used by CompiledGraphFunction.
     */
    public void markOutputs(INDArray[] outputs) {
        // No-op for now; end() will determine outputs from all lazy arrays
    }

    /**
     * End tracing: compile the traced graph and execute it.
     * All LazyINDArrays returned during tracing are materialized with real values.
     */
    public void end() {
        if (state != State.TRACING) {
            throw new IllegalStateException("GraphScope is not in TRACING state. Call begin() first.");
        }

        try {
            if (allLazyOutputs.isEmpty()) {
                log.debug("GraphScope: no ops traced, nothing to execute");
                state = State.EXECUTED;
                return;
            }

            // Identify leaf outputs (LazyINDArrays not consumed by other ops)
            List<LazyINDArray> leafOutputs = new ArrayList<>();
            for (LazyINDArray lazy : allLazyOutputs) {
                if (!consumedOutputs.contains(lazy)) {
                    leafOutputs.add(lazy);
                }
            }
            // If all are consumed, use all outputs
            if (leafOutputs.isEmpty()) {
                leafOutputs.addAll(allLazyOutputs);
            }

            // Collect output variable names
            Set<String> outputNames = new LinkedHashSet<>();
            for (LazyINDArray lazy : leafOutputs) {
                outputNames.add(lazy.getSdVariableName());
            }
            // Also add intermediate outputs that users might have references to
            for (LazyINDArray lazy : allLazyOutputs) {
                outputNames.add(lazy.getSdVariableName());
            }

            // Compute op sequence key for caching
            String newOpSequenceKey = computeOpSequenceKey();

            // Build placeholder map
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            for (Map.Entry<INDArray, String> entry : arrayToVar.entrySet()) {
                if (!(entry.getKey() instanceof LazyINDArray)) {
                    placeholders.put(entry.getValue(), entry.getKey());
                }
            }

            Map<String, INDArray> results;
            boolean cacheHit = newOpSequenceKey.equals(cachedOpSequenceKey)
                    && compiledPlan != null && executor != null;

            if (cacheHit) {
                // Replay with new inputs
                log.debug("GraphScope: replaying cached plan");
                results = executor.execute(compiledPlan, placeholders);
            } else {
                // Compile and execute
                log.debug("GraphScope: compiling new plan ({} ops)", opSequence.size());
                try {
                    ForwardExecutionDAGBuilder dagBuilder = new ForwardExecutionDAGBuilder(graph);
                    ForwardExecutionDAG dag = dagBuilder.buildForwardDAG(outputNames);
                    compiledPlan = DynamicShapePlanCompiler.compile(graph, dag, outputNames);

                    if (compiledPlan != null) {
                        executor = new DynamicShapePlanExecutor(graph, null);
                        results = executor.execute(compiledPlan, placeholders);
                        cachedOpSequenceKey = newOpSequenceKey;
                    } else {
                        // Fallback: compile returned null (control flow detected)
                        log.info("GraphScope: DSP compilation failed, falling back to eager execution");
                        results = executeEagerFallback(placeholders);
                    }
                } catch (Exception e) {
                    log.warn("GraphScope: compilation/execution failed, falling back to eager: {}",
                            e.getMessage());
                    results = executeEagerFallback(placeholders);
                }
            }

            // Materialize all lazy outputs
            if (results != null) {
                for (LazyINDArray lazy : allLazyOutputs) {
                    INDArray result = results.get(lazy.getSdVariableName());
                    if (result != null) {
                        lazy.materialize(result);
                    } else {
                        log.warn("GraphScope: no result for variable '{}'", lazy.getSdVariableName());
                    }
                }
            }

            state = State.EXECUTED;
            log.debug("GraphScope: execution complete, materialized {} outputs", allLazyOutputs.size());

        } finally {
            ACTIVE.remove();
        }
    }

    /**
     * Fallback: replay the traced ops eagerly when DSP compilation fails.
     */
    private Map<String, INDArray> executeEagerFallback(Map<String, INDArray> placeholders) {
        // Re-execute each op in the graph eagerly via the standard executor
        // This is a simple fallback that just runs the SameDiff graph
        try {
            // Use SameDiff's built-in execution as fallback
            Map<String, INDArray> results = graph.outputAll(placeholders);
            return results;
        } catch (Exception e) {
            log.error("GraphScope: eager fallback also failed", e);
            return Collections.emptyMap();
        }
    }

    private String computeOpSequenceKey() {
        StringBuilder sb = new StringBuilder();
        for (String opName : opSequence) {
            sb.append(opName).append(';');
        }
        return sb.toString();
    }

    @Override
    public void close() {
        if (state == State.TRACING) {
            ACTIVE.remove();
        }
        // Clean up compiled resources
        compiledPlan = null;
        executor = null;
        graph = null;
        arrayToVar = null;
        externalInputs = null;
        allLazyOutputs = null;
        consumedOutputs = null;
        opSequence = null;
        state = State.IDLE;
    }
}
