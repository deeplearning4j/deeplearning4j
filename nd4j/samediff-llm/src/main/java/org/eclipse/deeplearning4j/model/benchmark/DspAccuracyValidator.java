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

package org.eclipse.deeplearning4j.model.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.CapturingSlotInterceptor;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Core comparison engine for DSP accuracy validation.
 * Compares execution results between different execution modes to find the first divergent op.
 */
@Slf4j
public class DspAccuracyValidator {

    private final SameDiff model;
    private final ValidationConfig config;

    public DspAccuracyValidator(SameDiff model, ValidationConfig config) {
        this.model = model;
        this.config = config;
    }

    /**
     * Validates final outputs only: runs standard output() (DSP disabled) vs outputDirect(),
     * compares final outputs. Fast, low memory.
     *
     * @param placeholders input placeholders
     * @param outputNames  outputs to compare
     * @return divergence report
     */
    public DivergenceReport validateOutputs(Map<String, INDArray> placeholders, List<String> outputNames) {
        // Run reference: standard output() path
        boolean savedDsp = model.isDspAutoCompileEnabled();
        model.setDspAutoCompileEnabled(false);
        model.resetSession();
        Map<String, INDArray> refOutputs = model.output(placeholders, outputNames);
        // Dup reference outputs before they get closed
        Map<String, INDArray> refDuped = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> e : refOutputs.entrySet()) {
            refDuped.put(e.getKey(), e.getValue().dup());
        }

        // Run test: DSP-based outputDirect() path
        model.setDspAutoCompileEnabled(savedDsp);
        model.resetSession();
        Map<String, INDArray> testOutputs = model.outputDirect(placeholders,
                outputNames.toArray(new String[0]));

        // Compare
        List<OpDivergence> divergences = new ArrayList<>();
        int compared = 0;
        int matched = 0;
        for (String name : outputNames) {
            INDArray ref = refDuped.get(name);
            INDArray test = testOutputs.get(name);
            if (ref == null || test == null) {
                log.warn("Missing output for variable '{}': ref={}, test={}", name,
                        ref != null, test != null);
                continue;
            }
            compared++;
            double absTol = config.getAbsTolForOp("output");
            OpDivergence div = compareArrays(ref, test, name, "output", 0,
                    absTol, config.getDefaultRelTol(), null);
            if (div != null) {
                divergences.add(div);
                if (config.isStopAtFirst()) break;
                if (divergences.size() >= config.getMaxDivergences()) break;
            } else {
                matched++;
            }
        }

        // Clean up reference copies
        for (INDArray arr : refDuped.values()) {
            if (!arr.wasClosed()) arr.close();
        }

        return new DivergenceReport("standard", "dsp", compared, matched, divergences);
    }

    /**
     * Validates per-op: runs SLOT_BY_SLOT with interceptor as reference, then current mode
     * with interceptor as test. Compares every intermediate variable by name.
     *
     * @param placeholders input placeholders
     * @param outputNames  outputs to request
     * @return divergence report
     */
    public DivergenceReport validatePerOp(Map<String, INDArray> placeholders, List<String> outputNames) {
        // Reference run: SLOT_BY_SLOT
        CapturingSlotInterceptor refInterceptor = new CapturingSlotInterceptor();
        runWithInterceptor(placeholders, outputNames, GraphExecutionMode.SLOT_BY_SLOT,
                "SLOT_BY_SLOT (reference)", refInterceptor);

        // Test run: current configured mode
        CapturingSlotInterceptor testInterceptor = new CapturingSlotInterceptor();
        runWithInterceptor(placeholders, outputNames, null,
                "current mode (test)", testInterceptor);

        // Compare all captured variables
        Map<String, INDArray> refByName = refInterceptor.getByName();
        Map<String, INDArray> testByName = testInterceptor.getByName();

        List<OpDivergence> divergences = new ArrayList<>();
        int compared = 0;
        int matched = 0;

        for (Map.Entry<String, INDArray> refEntry : refByName.entrySet()) {
            String varName = refEntry.getKey();
            INDArray ref = refEntry.getValue();
            INDArray test = testByName.get(varName);
            if (test == null) {
                log.debug("Variable '{}' present in reference but not in test (may be fused)", varName);
                continue;
            }
            compared++;

            // Infer op name from variable name (convention: "opName" or "opName:N")
            String opName = varName.contains(":") ? varName.substring(0, varName.indexOf(':')) : varName;
            double absTol = config.getAbsTolForOp(opName);

            OpDivergence div = compareArrays(ref, test, varName, opName, 0,
                    absTol, config.getDefaultRelTol(), null);
            if (div != null) {
                divergences.add(div);
                if (config.isStopAtFirst()) break;
                if (divergences.size() >= config.getMaxDivergences()) break;
            } else {
                matched++;
            }
        }

        // Cleanup
        refInterceptor.clear();
        testInterceptor.clear();

        String testModeLabel = model.getGraphExecutionMode() != null
                ? model.getGraphExecutionMode().name() : "DSP_DEFAULT";
        return new DivergenceReport("SLOT_BY_SLOT", testModeLabel, compared, matched, divergences);
    }

    /**
     * Compares two arrays and returns an {@link OpDivergence} if they differ beyond tolerance.
     * Returns null if arrays match within tolerance.
     */
    public static OpDivergence compareArrays(INDArray ref, INDArray test, String varName,
                                              String opName, int stepIdx,
                                              double absTol, double relTol,
                                              String[] inputVarNames) {
        // Shape check
        if (!Arrays.equals(ref.shape(), test.shape())) {
            return new OpDivergence(stepIdx, opName, varName, ref.shape(), ref.dataType(),
                    Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE,
                    0, 0, 0, inputVarNames);
        }

        // DataType check — only warn, still compare values
        if (ref.dataType() != test.dataType()) {
            log.warn("Type mismatch for '{}': ref={}, test={}", varName, ref.dataType(), test.dataType());
        }

        // Skip non-floating-point comparisons (int indices, bool masks)
        if (!ref.dataType().isFPType()) {
            // For integer types, do exact comparison
            if (ref.equalsWithEps(test, 0)) {
                return null;
            }
            // Find first difference
            INDArray diff = ref.neq(test).castTo(DataType.FLOAT);
            long numDiff = (long) diff.sumNumber().doubleValue();
            if (!diff.wasClosed()) diff.close();
            return new OpDivergence(stepIdx, opName, varName, ref.shape(), ref.dataType(),
                    numDiff, numDiff, 1.0, 0, 0, 0, inputVarNames);
        }

        // Cast to float64 for comparison
        INDArray refF = ref.dataType() == DataType.DOUBLE ? ref : ref.castTo(DataType.DOUBLE);
        INDArray testF = test.dataType() == DataType.DOUBLE ? test : test.castTo(DataType.DOUBLE);

        // NaN/Inf check
        boolean refHasNan = Nd4j.getExecutioner().execAndReturn(
                new org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN(refF)).z().getDouble(0) > 0;
        boolean testHasNan = Nd4j.getExecutioner().execAndReturn(
                new org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN(testF)).z().getDouble(0) > 0;
        if (refHasNan != testHasNan) {
            return new OpDivergence(stepIdx, opName, varName, ref.shape(), ref.dataType(),
                    Double.NaN, Double.NaN, Double.NaN,
                    0, 0, 0, inputVarNames);
        }

        // Compute absolute difference
        INDArray absDiff = Nd4j.math.abs(refF.sub(testF));
        double maxAbsDiff = absDiff.maxNumber().doubleValue();
        double meanAbsDiff = absDiff.meanNumber().doubleValue();

        // Check against tolerance
        if (maxAbsDiff <= absTol) {
            if (refF != ref && !refF.wasClosed()) refF.close();
            if (testF != test && !testF.wasClosed()) testF.close();
            if (!absDiff.wasClosed()) absDiff.close();
            return null;
        }

        // Compute relative difference
        INDArray refAbs = Nd4j.math.abs(refF);
        INDArray denominator = Nd4j.math.max(refAbs, Nd4j.scalar(DataType.DOUBLE, 1e-8));
        INDArray relDiff = absDiff.div(denominator);
        double maxRelDiff = relDiff.maxNumber().doubleValue();

        // If both abs and rel thresholds are exceeded, it's a divergence
        if (maxAbsDiff > absTol && maxRelDiff > relTol) {
            long maxIdx = Nd4j.argMax(absDiff.reshape(1, -1), 1).getLong(0);
            double refVal = refF.getDouble(maxIdx);
            double testVal = testF.getDouble(maxIdx);

            // Cleanup
            if (refF != ref && !refF.wasClosed()) refF.close();
            if (testF != test && !testF.wasClosed()) testF.close();
            if (!absDiff.wasClosed()) absDiff.close();
            if (!relDiff.wasClosed()) relDiff.close();
            if (!refAbs.wasClosed()) refAbs.close();
            if (!denominator.wasClosed()) denominator.close();

            return new OpDivergence(stepIdx, opName, varName, ref.shape(), ref.dataType(),
                    maxAbsDiff, meanAbsDiff, maxRelDiff,
                    maxIdx, refVal, testVal, inputVarNames);
        }

        // Cleanup
        if (refF != ref && !refF.wasClosed()) refF.close();
        if (testF != test && !testF.wasClosed()) testF.close();
        if (!absDiff.wasClosed()) absDiff.close();
        if (!relDiff.wasClosed()) relDiff.close();
        if (!refAbs.wasClosed()) refAbs.close();
        if (!denominator.wasClosed()) denominator.close();

        return null;
    }

    /**
     * Runs the model with the given execution mode and captures slot outputs via interceptor.
     */
    private void runWithInterceptor(Map<String, INDArray> placeholders, List<String> outputNames,
                                    GraphExecutionMode mode, String label,
                                    CapturingSlotInterceptor interceptor) {
        model.resetSession();

        if (mode != null) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            model.compileNativeDynamicShapePlan(outputNames, mode, true);
        }

        // Phase 1: warm-up execution to initialize session and executor
        model.outputDirect(placeholders, outputNames.toArray(new String[0]));

        // Phase 2: attach interceptor to the now-existing executor
        InferenceSession session = model.getOrCreateSession();
        if (session != null) {
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            if (executor != null) {
                executor.setSlotOutputInterceptor(interceptor);
            } else {
                log.warn("{}: no DynamicShapePlanExecutor found in session", label);
            }
        }

        // Phase 3: run again with interceptor active
        model.outputDirect(placeholders, outputNames.toArray(new String[0]));

        // Phase 4: detach interceptor
        if (session != null) {
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            if (executor != null) {
                executor.setSlotOutputInterceptor(null);
            }
        }

        log.info("{}: captured {} variables", label, interceptor.getByName().size());
    }
}
