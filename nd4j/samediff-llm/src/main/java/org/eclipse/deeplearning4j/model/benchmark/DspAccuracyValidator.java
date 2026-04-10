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
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Compares DSP execution outputs against a reference (standard op-by-op) path
 * to identify the exact source of inaccuracies.
 *
 * <h3>Usage modes</h3>
 * <ul>
 *   <li>{@link #validateOutputs} — compares final outputs only (fast, low memory)</li>
 *   <li>{@link #validatePerOp} — compares every intermediate variable (finds first divergent op)</li>
 *   <li>{@link #compareOutputMaps} — static utility for comparing two output maps</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>
 *   DspAccuracyValidator validator = new DspAccuracyValidator(model, ValidationConfig.standard());
 *   DivergenceReport report = validator.validateOutputs(placeholders, "logits");
 *   System.out.println(report.toReport());
 * </pre>
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
     * Compares final outputs: standard output() vs DSP outputDirect().
     *
     * <p>Runs the reference path with DSP disabled (standard InferenceSession),
     * then runs the DSP path. Compares only the requested output variables.</p>
     *
     * @param placeholders input placeholder values
     * @param outputNames  output variable names to compare
     * @return divergence report
     */
    public DivergenceReport validateOutputs(Map<String, INDArray> placeholders, String... outputNames) {
        Map<String, INDArray> reference = runStandard(placeholders, outputNames);
        Map<String, INDArray> dspResult = runDsp(placeholders, outputNames);
        try {
            return compareOutputMaps(reference, dspResult, "output()", "outputDirect()", config);
        } finally {
            closeAll(reference);
            closeAll(dspResult);
        }
    }

    /**
     * Per-op comparison using slot interceptors.
     *
     * <p>Runs DSP in SLOT_BY_SLOT mode (reference) with a capturing interceptor,
     * then runs DSP in the model's current execution mode (test) with another
     * capturing interceptor. Compares every captured intermediate variable.</p>
     *
     * <p>This identifies the first op where the two execution paths diverge,
     * which is the root cause of any final-output difference.</p>
     *
     * @param placeholders input placeholder values
     * @param outputNames  output variable names (determines the plan)
     * @return divergence report with per-op detail
     */
    public DivergenceReport validatePerOp(Map<String, INDArray> placeholders, String... outputNames) {
        throw new UnsupportedOperationException(
                "Per-op validation is no longer available. Native C++ execution does not support " +
                "Java-side slot interceptor callbacks. Use validateOutputs() for output-level comparison, " +
                "or use C++ diagnostics (dspDiagGetPlanReport) for per-slot inspection.");
    }

    /**
     * Compares two output maps variable-by-variable.
     *
     * @param reference    reference outputs (ground truth)
     * @param test         test outputs to validate
     * @param refLabel     label for reference mode (e.g. "output()")
     * @param testLabel    label for test mode (e.g. "outputDirect()")
     * @param config       tolerance configuration
     * @return divergence report
     */
    public static DivergenceReport compareOutputMaps(Map<String, INDArray> reference,
                                                      Map<String, INDArray> test,
                                                      String refLabel, String testLabel,
                                                      ValidationConfig config) {
        List<OpDivergence> divergences = new ArrayList<>();
        int compared = 0;
        int matched = 0;

        for (Map.Entry<String, INDArray> entry : reference.entrySet()) {
            String varName = entry.getKey();
            INDArray refArr = entry.getValue();
            INDArray testArr = test.get(varName);

            if (testArr == null) {
                divergences.add(new OpDivergence(
                        -1, "MISSING", varName,
                        refArr.shape(), refArr.dataType().toString(),
                        Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE,
                        0, 0, 0, null,
                        config.getDefaultAbsTol()));
                compared++;
                continue;
            }

            compared++;
            double absTol = config.getDefaultAbsTol();
            OpDivergence div = compareArrays(refArr, testArr, varName, "final_output", -1,
                    absTol, config.getDefaultRelTol(), null);
            if (div == null) {
                matched++;
            } else {
                divergences.add(div);
                if (config.isStopAtFirst()) break;
                if (divergences.size() >= config.getMaxDivergences()) break;
            }
        }

        return new DivergenceReport(refLabel, testLabel, compared, matched, divergences);
    }

    /**
     * Element-wise comparison of two INDArrays. Returns null if within tolerance.
     *
     * <p>Comparison order: shape check → dtype check → NaN/Inf check → absolute diff →
     * relative diff. Both absolute AND relative must exceed their tolerances for a
     * divergence to be reported (i.e., a large absolute diff on a large value is OK
     * if the relative diff is small).</p>
     *
     * @param reference     reference array (ground truth)
     * @param test          test array to validate
     * @param varName       variable name for reporting
     * @param opName        op name for reporting
     * @param stepIdx       step index for reporting (-1 for final outputs)
     * @param absTol        absolute tolerance threshold
     * @param relTol        relative tolerance threshold
     * @param inputVarNames input variable names for tracing (may be null)
     * @return OpDivergence if arrays differ beyond tolerance, null if they match
     */
    public static OpDivergence compareArrays(INDArray reference, INDArray test,
                                              String varName, String opName, int stepIdx,
                                              double absTol, double relTol,
                                              String[] inputVarNames) {
        // Shape mismatch
        if (!Arrays.equals(reference.shape(), test.shape())) {
            return new OpDivergence(stepIdx, opName, varName,
                    reference.shape(), reference.dataType().toString(),
                    Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE,
                    0, 0, 0, inputVarNames, absTol);
        }

        // Skip non-numeric types
        DataType dt = reference.dataType();
        if (!dt.isFPType()) {
            // For integer/bool types, exact match
            if (reference.equals(test)) return null;
            long length = reference.length();
            INDArray refFlat = reference.reshape(length);
            INDArray testFlat = test.reshape(length);
            for (long i = 0; i < Math.min(length, 1000); i++) {
                if (refFlat.getLong(i) != testFlat.getLong(i)) {
                    return new OpDivergence(stepIdx, opName, varName,
                            reference.shape(), dt.toString(),
                            1, 1, 1,
                            i, refFlat.getDouble(i), testFlat.getDouble(i),
                            inputVarNames, 0);
                }
            }
            return null;
        }

        // Cast to FLOAT for comparison if needed (FP16 sub() can lose precision)
        INDArray refFloat = dt == DataType.FLOAT ? reference : reference.castTo(DataType.FLOAT);
        INDArray testFloat = dt == DataType.FLOAT ? test : test.castTo(DataType.FLOAT);

        // Absolute difference
        INDArray diff = refFloat.sub(testFloat);
        INDArray absDiff = org.nd4j.linalg.ops.transforms.Transforms.abs(diff);
        double maxAbs = absDiff.maxNumber().doubleValue();
        double meanAbs = absDiff.meanNumber().doubleValue();

        // Fast path: within absolute tolerance
        if (maxAbs <= absTol) {
            closeTempArrays(diff, absDiff, refFloat, testFloat, reference, test, dt);
            return null;
        }

        // Relative difference for non-zero elements
        INDArray refAbs = org.nd4j.linalg.ops.transforms.Transforms.abs(refFloat).addi(1e-12);
        INDArray relDiff = absDiff.div(refAbs);
        double maxRel = relDiff.maxNumber().doubleValue();

        // Pass if relative tolerance satisfied (large values may exceed absolute tol)
        if (maxRel <= relTol) {
            closeTempArrays(diff, absDiff, refFloat, testFloat, reference, test, dt);
            refAbs.close();
            relDiff.close();
            return null;
        }

        // Find the element with maximum absolute difference
        long flatLength = absDiff.length();
        INDArray flatDiff = absDiff.reshape(1, flatLength);
        long maxIdx = Nd4j.argMax(flatDiff, 1).getLong(0);
        INDArray refFlat = refFloat.reshape(flatLength);
        INDArray testFlat = testFloat.reshape(flatLength);
        double refVal = refFlat.getDouble(maxIdx);
        double testVal = testFlat.getDouble(maxIdx);

        closeTempArrays(diff, absDiff, refFloat, testFloat, reference, test, dt);
        refAbs.close();
        relDiff.close();

        return new OpDivergence(stepIdx, opName, varName,
                reference.shape(), dt.toString(),
                maxAbs, meanAbs, maxRel,
                maxIdx, refVal, testVal,
                inputVarNames, absTol);
    }

    // -- Internal helpers --

    private Map<String, INDArray> runStandard(Map<String, INDArray> placeholders, String[] outputNames) {
        boolean wasDspEnabled = model.isDspAutoCompileEnabled();
        model.setDspAutoCompileEnabled(false);
        model.resetSession();
        try {
            Map<String, INDArray> result = model.output(placeholders, outputNames);
            // Dup all outputs so they survive session reset
            Map<String, INDArray> duped = new LinkedHashMap<>();
            for (Map.Entry<String, INDArray> e : result.entrySet()) {
                duped.put(e.getKey(), e.getValue().dup());
            }
            return duped;
        } finally {
            model.setDspAutoCompileEnabled(wasDspEnabled);
        }
    }

    private Map<String, INDArray> runDsp(Map<String, INDArray> placeholders, String[] outputNames) {
        model.resetSession();
        model.setDspAutoCompileEnabled(true);
        Map<String, INDArray> result = model.outputDirect(placeholders, outputNames);
        Map<String, INDArray> duped = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> e : result.entrySet()) {
            duped.put(e.getKey(), e.getValue().dup());
        }
        return duped;
    }

    private static void closeTempArrays(INDArray diff, INDArray absDiff,
                                         INDArray refFloat, INDArray testFloat,
                                         INDArray origRef, INDArray origTest,
                                         DataType origDt) {
        diff.close();
        absDiff.close();
        if (origDt != DataType.FLOAT) {
            refFloat.close();
            testFloat.close();
        }
    }

    private static void closeAll(Map<String, INDArray> map) {
        for (INDArray arr : map.values()) {
            if (arr != null && !arr.wasClosed()) {
                arr.close();
            }
        }
    }
}
