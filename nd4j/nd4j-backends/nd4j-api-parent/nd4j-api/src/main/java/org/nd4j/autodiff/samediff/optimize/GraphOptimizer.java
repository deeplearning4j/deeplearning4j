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

package org.nd4j.autodiff.samediff.optimize;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.debug.OptimizationDebugger;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;

import java.util.*;

/**
 * Graph optimizer for SameDiff graphs.
 * Applies a series of optimization passes to reduce graph complexity and improve performance.
 *
 * @author Alex Black
 */
@Slf4j
public class GraphOptimizer {

    /**
     * Maximum number of optimization iterations. Can be overridden via system property.
     */
    private static final int MAX_ITERATIONS = Integer.getInteger("nd4j.optimizer.maxIterations", 3);

    /**
     * Whether to log each applied optimization. Can be disabled for performance.
     */
    private static final boolean LOG_APPLIED_OPTS = Boolean.getBoolean("nd4j.optimizer.logApplied");

    /**
     * Comma-separated list of optimizer class simple names to skip.
     * E.g. -Dnd4j.optimizer.skip=NormalizationFusionOptimizations,QuantizationOptimizations
     */
    private static final Set<String> SKIP_OPTIMIZERS;
    static {
        String skipProp = System.getProperty("nd4j.optimizer.skip", "");
        Set<String> s = new HashSet<>();
        if (!skipProp.isEmpty()) {
            for (String name : skipProp.split(",")) {
                s.add(name.trim());
            }
        }
        SKIP_OPTIMIZERS = s;
    }

    public static List<OptimizerSet> defaultOptimizations() {
        List<OptimizerSet> all = Arrays.<OptimizerSet>asList(
                new UnusedFunctionOptimizations(),
                new ConstantFunctionOptimizations(),
                new AlgebraicOptimizations(),        // x+0->x, x*1->x, x*0->0, etc.
                new IdentityFunctionOptimizations(),
                new ShapeFunctionOptimizations(),
                new ActivationFusionOptimizations(), // sigmoid(x)*x -> swish, SwiGLU detection
                new NormalizationFusionOptimizations(), // RMSNorm detection
                new GatedDeltaNetFusionOptimizations(), // GDN pattern fusion
                new LinearFusionOptimizations(),
                new AttentionFusionOptimizations(),  // Fuse attention patterns
                new QuantizationOptimizations(),     // Remove redundant casts, FP16 quantization
                new UnusedFunctionOptimizations(),
                new CuDNNFunctionOptimizations()
        );
        if (SKIP_OPTIMIZERS.isEmpty()) {
            return all;
        }
        List<OptimizerSet> filtered = new ArrayList<>();
        for (OptimizerSet opt : all) {
            if (SKIP_OPTIMIZERS.contains(opt.getClass().getSimpleName())) {
                log.info("Skipping optimizer: {}", opt.getClass().getSimpleName());
            } else {
                filtered.add(opt);
            }
        }
        return filtered;
    }

    public static SameDiff optimize(SameDiff graph, String... requiredOutputs){
        return optimize(graph, Arrays.asList(requiredOutputs));
    }

    public static SameDiff optimize(SameDiff graph, List<String> requiredOutputs){
        return optimize(graph, requiredOutputs, defaultOptimizations());
    }

    public static SameDiff optimize(SameDiff graph, List<String> requiredOutputs, List<OptimizerSet> optimizations) {
        return optimize(graph, requiredOutputs, optimizations, null);
    }

    public static SameDiff optimize(SameDiff graph, List<String> requiredOutputs, List<OptimizerSet> optimizations, OptimizationDebugger debugger){
        long startTime = System.currentTimeMillis();

        // Use full dup() - shallowClone shares DifferentialFunction objects which corrupts the original
        SameDiff sd = graph.dup();

        // dup() via SDNB round-trip may not preserve graph-level outputs.
        // Restore them so fusion optimizers can detect which variables are outputs.
        if ((sd.outputs() == null || sd.outputs().isEmpty()) && graph.outputs() != null && !graph.outputs().isEmpty()) {
            List<String> validOutputs = new ArrayList<>();
            for (String out : graph.outputs()) {
                if (sd.hasVariable(out)) {
                    validOutputs.add(out);
                }
            }
            if (!validOutputs.isEmpty()) {
                sd.setOutputs(validOutputs);
            }
        }

        ArrayHolder cArr = sd.getConstantArrays();
        ArrayHolder vArr = sd.getVariablesArrays();

        OptimizationHelper h = new OptimizationHelper(sd, new OptimizationConfig());
        // Initialize fast HashMap caches for O(1) lookups instead of PatriciaTrie O(k)
        h.initializeCaches(sd);

        // Pre-collect all optimizers once to avoid repeated reflection calls
        List<Optimizer> allOptimizers = new ArrayList<>();
        for (OptimizerSet s : optimizations) {
            allOptimizers.addAll(s.getOptimizers());
        }

        // Pre-compute op type filters for each optimizer
        Map<Optimizer, Set<Class<? extends DifferentialFunction>>> optimizerFilters = new HashMap<>();
        for (Optimizer o : allOptimizers) {
            Set<Class<? extends DifferentialFunction>> applicableTypes = o.getApplicableOpTypes();
            if (applicableTypes != null && !applicableTypes.isEmpty()) {
                optimizerFilters.put(o, applicableTypes);
            }
        }

        log.debug("Running {} optimizers over {} ops ({} with type filters)",
                allOptimizers.size(), sd.getOps().size(), optimizerFilters.size());

        // Graph-level DCE: backward reachability sweep from required outputs.
        // Removes ops (and their ARRAY output variables) that cannot reach any output.
        // Placeholders, constants, and variables are never removed — only ARRAY vars
        // produced by unreachable ops.
        int dceRemoved = 0;
        if (requiredOutputs != null && !requiredOutputs.isEmpty()) {
            Set<String> reachableOps = new HashSet<>();
            Set<String> reachableVars = new HashSet<>();
            Deque<String> varQueue = new ArrayDeque<>(requiredOutputs);

            // Walk backward from outputs
            while (!varQueue.isEmpty()) {
                String varName = varQueue.poll();
                if (!reachableVars.add(varName)) continue;

                Variable v = sd.getVariables().get(varName);
                if (v == null) continue;

                String producerOp = v.getOutputOfOp();
                if (producerOp == null) continue; // placeholder/constant/variable — leaf
                if (!reachableOps.add(producerOp)) continue;

                SameDiffOp op = sd.getOps().get(producerOp);
                if (op == null) continue;

                // Enqueue all inputs of this op
                if (op.getInputsToOp() != null) {
                    varQueue.addAll(op.getInputsToOp());
                }
                // Also follow control dependencies
                if (op.getControlDeps() != null) {
                    varQueue.addAll(op.getControlDeps());
                }
            }

            // Collect unreachable ops
            List<String> opsToRemove = new ArrayList<>();
            for (String opName : sd.getOps().keySet()) {
                if (!reachableOps.contains(opName)) {
                    opsToRemove.add(opName);
                }
            }

            // Remove unreachable ops and their ARRAY output variables
            for (String opName : opsToRemove) {
                SameDiffOp op = sd.getOps().get(opName);
                if (op == null) continue;

                // Remove output variables that are ARRAY type (produced by this op)
                if (op.getOutputsOfOp() != null) {
                    for (String outVar : op.getOutputsOfOp()) {
                        Variable v = sd.getVariables().get(outVar);
                        if (v != null && v.getVariable() != null
                                && v.getVariable().getVariableType() == VariableType.ARRAY) {
                            OptimizationUtils.removeVariable(sd, h, outVar);
                        }
                    }
                }
                OptimizationUtils.removeOp(sd, h, opName);
                dceRemoved++;
            }

            if (dceRemoved > 0) {
                log.info("DCE: removed {} unreachable ops (from {} total)", dceRemoved, dceRemoved + sd.getOps().size());
            }
        }

        int totalApplied = dceRemoved;
        int totalSkipped = 0;

        // Run multiple iterations - some optimizations enable others
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            int appliedThisIteration = 0;

            for (Optimizer o : allOptimizers) {
                Set<Class<? extends DifferentialFunction>> filter = optimizerFilters.get(o);

                // Get current ops snapshot once per optimizer
                String[] opNames = sd.getOps().keySet().toArray(new String[0]);

                for (String opName : opNames) {
                    // Check if op still exists (may have been removed by previous optimization)
                    SameDiffOp op = sd.getOps().get(opName);
                    if (op == null)
                        continue;

                    // Fast path: skip ops that don't match the optimizer's type filter
                    if (filter != null) {
                        DifferentialFunction opFunc = op.getOp();
                        if (opFunc != null) {
                            boolean matches = false;
                            Class<?> opClass = opFunc.getClass();
                            for (Class<? extends DifferentialFunction> filterClass : filter) {
                                if (filterClass.isAssignableFrom(opClass)) {
                                    matches = true;
                                    break;
                                }
                            }
                            if (!matches) {
                                totalSkipped++;
                                continue;
                            }
                        }
                    }

                    if (debugger != null)
                        debugger.beforeOptimizationCheck(sd, op, o);

                    boolean applied = o.checkAndApply(sd, h, op, cArr, vArr);
                    if (applied) {
                        appliedThisIteration++;
                        if (LOG_APPLIED_OPTS) {
                            log.info("Applied {} to op {}", o.getClass().getSimpleName(), opName);
                        }
                    }

                    if (debugger != null)
                        debugger.afterOptimizationsCheck(sd, op, o, applied);
                }
            }

            totalApplied += appliedThisIteration;

            // Early exit if no optimizations were applied this iteration
            if (appliedThisIteration == 0) {
                log.debug("No optimizations applied in iteration {}, stopping early", iteration);
                break;
            }
        }

        log.debug("Skipped {} op checks due to type filtering", totalSkipped);

        long elapsed = System.currentTimeMillis() - startTime;

        // Count variable types for logging
        int constBefore = 0, constAfter = 0;
        int varBefore = 0, varAfter = 0;
        int arrBefore = 0, arrAfter = 0;

        for (SDVariable v : graph.variables()) {
            switch (v.getVariableType()) {
                case VARIABLE: varBefore++; break;
                case CONSTANT: constBefore++; break;
                case ARRAY: arrBefore++; break;
                case PLACEHOLDER: break;
            }
        }

        for (SDVariable v : sd.variables()) {
            switch (v.getVariableType()) {
                case VARIABLE: varAfter++; break;
                case CONSTANT: constAfter++; break;
                case ARRAY: arrAfter++; break;
                case PLACEHOLDER: break;
            }
        }

        log.info("Optimization completed in {}ms, {} optimizations applied", elapsed, totalApplied);
        log.info("Total variables: {} before, {} after", graph.getVariables().size(), sd.getVariables().size());
        log.info("Constant variables: {} before, {} after", constBefore, constAfter);
        log.info("Array type variables: {} before, {} after", arrBefore, arrAfter);
        log.info("Variable type variables: {} before, {} after", varBefore, varAfter);
        log.info("Ops: {} before, {} after", graph.getOps().size(), sd.getOps().size());

        // Validate the optimized graph: every op's inputs must reference existing variables
        List<String> validationErrors = validateGraph(sd);
        if (!validationErrors.isEmpty()) {
            log.warn("Graph optimization produced {} validation errors — returning original unoptimized graph", validationErrors.size());
            for (String err : validationErrors) {
                log.warn("  Validation error: {}", err);
            }
            // Return a dup of the original graph instead of the corrupted optimized one
            SameDiff fallback = graph.dup();
            if (requiredOutputs != null && !requiredOutputs.isEmpty()) {
                fallback.setOutputs(new ArrayList<>(requiredOutputs));
            } else if (graph.outputs() != null && !graph.outputs().isEmpty()) {
                fallback.setOutputs(new ArrayList<>(graph.outputs()));
            }
            return fallback;
        }

        // Preserve outputs: fusion optimizers may have redirected output names
        // (e.g. lm_logits -> rms_norm_linear). Prefer sd.outputs() which reflects
        // those redirections, then validate against surviving variables.
        List<String> currentOutputs = sd.outputs();
        if (currentOutputs != null && !currentOutputs.isEmpty()) {
            // sd.outputs() was set during optimization — trust redirections from fusion ops
            List<String> surviving = new ArrayList<>();
            for (String name : currentOutputs) {
                if (sd.hasVariable(name)) {
                    surviving.add(name);
                } else {
                    log.debug("Output variable '{}' removed by optimization — dropping from output list", name);
                }
            }
            if (!surviving.isEmpty()) {
                sd.setOutputs(surviving);
            }
        } else if (requiredOutputs != null && !requiredOutputs.isEmpty()) {
            List<String> surviving = new ArrayList<>();
            for (String name : requiredOutputs) {
                if (sd.hasVariable(name)) {
                    surviving.add(name);
                } else {
                    log.debug("Output variable '{}' removed by optimization — dropping from output list", name);
                }
            }
            if (!surviving.isEmpty()) {
                sd.setOutputs(surviving);
            }
        } else if (graph.outputs() != null && !graph.outputs().isEmpty()) {
            List<String> surviving = new ArrayList<>();
            for (String name : graph.outputs()) {
                if (sd.hasVariable(name)) {
                    surviving.add(name);
                } else {
                    log.debug("Output variable '{}' removed by optimization — dropping from output list", name);
                }
            }
            if (!surviving.isEmpty()) {
                sd.setOutputs(surviving);
            }
        }

        return sd;
    }

    /**
     * Validate the optimized graph for structural integrity.
     * Checks that every op's input variables exist in the variables map.
     *
     * @return list of validation error messages (empty if valid)
     */
    private static List<String> validateGraph(SameDiff sd) {
        List<String> errors = new ArrayList<>();
        Map<String, Variable> vars = sd.getVariables();
        for (Map.Entry<String, SameDiffOp> entry : sd.getOps().entrySet()) {
            String opName = entry.getKey();
            SameDiffOp op = entry.getValue();
            List<String> inputs = op.getInputsToOp();
            if (inputs == null) continue;
            for (int i = 0; i < inputs.size(); i++) {
                String inputVar = inputs.get(i);
                if (!vars.containsKey(inputVar)) {
                    errors.add("Op '" + opName + "' input[" + i + "] references missing variable '" + inputVar + "'");
                }
            }
        }
        return errors;
    }

    /**
     * Creates a shallow clone of a SameDiff graph suitable for optimization.
     * This clones the graph structure (ops, variables, their connections) but SHARES
     * the underlying array data, which is the expensive part.
     *
     * This is much faster than SameDiff.dup() which serializes/deserializes
     * everything through FlatBuffers.
     *
     * @param original The original SameDiff graph
     * @return A shallow clone suitable for optimization
     */
    private static SameDiff shallowCloneForOptimization(SameDiff original) {
        long start = System.currentTimeMillis();

        SameDiff clone = SameDiff.create();

        // Share the array holders - these contain the actual heavy data
        // The arrays themselves are not modified during optimization, only the graph structure
        clone.setConstantArrays(original.getConstantArrays());
        clone.setVariablesArrays(original.getVariablesArrays());
        clone.setEagerArrays(original.getEagerArrays());

        // Clone variables map - need new Variable instances with copied lists
        // but the SDVariable references can be updated to point to the new SameDiff
        Map<String, Variable> originalVars = original.getVariables();
        for (Map.Entry<String, Variable> entry : originalVars.entrySet()) {
            Variable origVar = entry.getValue();

            // Create new SDVariable pointing to the clone
            SDVariable origSDVar = origVar.getVariable();
            SDVariable clonedSDVar = new SDVariable(
                    origSDVar.name(),
                    origSDVar.getVariableType(),
                    clone,
                    origSDVar.getShape(),
                    origSDVar.dataType()
            );

            // Create new Variable with copied lists
            Variable clonedVar = Variable.builder()
                    .name(origVar.getName())
                    .variable(clonedSDVar)
                    .inputsForOp(origVar.getInputsForOp() != null ? new ArrayList<>(origVar.getInputsForOp()) : null)
                    .controlDepsForOp(origVar.getControlDepsForOp() != null ? new ArrayList<>(origVar.getControlDepsForOp()) : null)
                    .controlDepsForVar(origVar.getControlDepsForVar() != null ? new ArrayList<>(origVar.getControlDepsForVar()) : null)
                    .outputOfOp(origVar.getOutputOfOp())
                    .controlDeps(origVar.getControlDeps() != null ? new ArrayList<>(origVar.getControlDeps()) : null)
                    .build();

            clone.getVariables().put(entry.getKey(), clonedVar);
        }

        // Clone ops map - need new SameDiffOp instances with copied lists
        Map<String, SameDiffOp> originalOps = original.getOps();
        for (Map.Entry<String, SameDiffOp> entry : originalOps.entrySet()) {
            SameDiffOp origOp = entry.getValue();

            // Clone the DifferentialFunction and set the new SameDiff reference
            // The op itself is lightweight - it's just metadata about the operation
            SameDiffOp clonedOp = SameDiffOp.builder()
                    .name(origOp.getName())
                    .op(origOp.getOp())  // Share the op - it contains no arrays
                    .inputsToOp(origOp.getInputsToOp() != null ? new ArrayList<>(origOp.getInputsToOp()) : null)
                    .outputsOfOp(origOp.getOutputsOfOp() != null ? new ArrayList<>(origOp.getOutputsOfOp()) : null)
                    .controlDeps(origOp.getControlDeps() != null ? new ArrayList<>(origOp.getControlDeps()) : null)
                    .varControlDeps(origOp.getVarControlDeps() != null ? new ArrayList<>(origOp.getVarControlDeps()) : null)
                    .controlDepFor(origOp.getControlDepFor() != null ? new ArrayList<>(origOp.getControlDepFor()) : null)
                    .build();

            // Update the op's SameDiff reference
            if (clonedOp.getOp() != null) {
                clonedOp.getOp().setSameDiff(clone);
            }

            clone.getOps().put(entry.getKey(), clonedOp);
        }

        // Copy loss variables
        for (String lossVar : original.getLossVariables()) {
            clone.addLossVariable(lossVar);
        }

        // Copy outputs if set
        if (original.outputs() != null) {
            clone.setOutputs(new ArrayList<>(original.outputs()));
        }

        long elapsed = System.currentTimeMillis() - start;
        log.debug("Shallow clone created in {}ms ({} vars, {} ops)",
                elapsed, clone.getVariables().size(), clone.getOps().size());

        return clone;
    }

}
