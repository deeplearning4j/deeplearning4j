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

import java.util.HashMap;
import java.util.Map;

/**
 * Configures tolerance thresholds and comparison behavior for DSP accuracy validation.
 *
 * <p>Different ops accumulate floating-point error at different rates. matmul with TF32
 * tensor cores introduces ~1e-3 relative error per multiply-add; elementwise ops are
 * exact for the same datatype. This class lets callers specify per-op-type tolerances.</p>
 */
public class ValidationConfig {

    private double defaultAbsTol = 1e-4;
    private double defaultRelTol = 1e-3;
    private Map<String, Double> opAbsTol = new HashMap<>();
    private boolean stopAtFirst = false;
    private int maxDivergences = 100;
    private boolean captureInputsForDivergent = false;
    private double logitAbsTol = 1e-3;
    private int topKOverlap = 10;

    /**
     * Standard tolerances: matmul/GEMM at 1e-2, norm ops at 5e-4, everything else at 1e-4.
     * Suitable for TF32-enabled execution.
     */
    public static ValidationConfig standard() {
        ValidationConfig c = new ValidationConfig();
        c.opAbsTol.put("matmul", 1e-2);
        c.opAbsTol.put("mmul", 1e-2);
        c.opAbsTol.put("batched_gemm", 1e-2);
        c.opAbsTol.put("fused_gemm_swiglu", 1e-2);
        c.opAbsTol.put("softmax", 5e-4);
        c.opAbsTol.put("layer_norm", 5e-4);
        c.opAbsTol.put("rms_norm", 5e-4);
        return c;
    }

    /**
     * Strict tolerances for FP32 without TF32. Catches precision bugs that TF32 masks.
     */
    public static ValidationConfig strict() {
        ValidationConfig c = new ValidationConfig();
        c.defaultAbsTol = 1e-6;
        c.defaultRelTol = 1e-5;
        c.opAbsTol.put("matmul", 1e-5);
        c.opAbsTol.put("mmul", 1e-5);
        c.opAbsTol.put("batched_gemm", 1e-5);
        c.opAbsTol.put("softmax", 1e-6);
        return c;
    }

    /**
     * Relaxed tolerances for TF32-enabled configs where cuBLAS may produce large deltas.
     */
    public static ValidationConfig tf32Tolerant() {
        ValidationConfig c = new ValidationConfig();
        c.defaultAbsTol = 1e-2;
        c.defaultRelTol = 5e-2;
        c.opAbsTol.put("matmul", 5e-2);
        c.opAbsTol.put("mmul", 5e-2);
        c.opAbsTol.put("batched_gemm", 5e-2);
        c.opAbsTol.put("fused_gemm_swiglu", 5e-2);
        c.logitAbsTol = 1e-1;
        return c;
    }

    /**
     * Returns the absolute tolerance for a given op name.
     * Falls back to {@link #getDefaultAbsTol()} if no per-op override exists.
     */
    public double getAbsTolForOp(String opName) {
        Double override = opAbsTol.get(opName);
        return override != null ? override : defaultAbsTol;
    }

    // Fluent setters
    public ValidationConfig defaultAbsTol(double v) { this.defaultAbsTol = v; return this; }
    public ValidationConfig defaultRelTol(double v) { this.defaultRelTol = v; return this; }
    public ValidationConfig opAbsTol(String op, double tol) { this.opAbsTol.put(op, tol); return this; }
    public ValidationConfig stopAtFirst(boolean v) { this.stopAtFirst = v; return this; }
    public ValidationConfig maxDivergences(int v) { this.maxDivergences = v; return this; }
    public ValidationConfig captureInputsForDivergent(boolean v) { this.captureInputsForDivergent = v; return this; }
    public ValidationConfig logitAbsTol(double v) { this.logitAbsTol = v; return this; }
    public ValidationConfig topKOverlap(int v) { this.topKOverlap = v; return this; }

    // Getters
    public double getDefaultAbsTol() { return defaultAbsTol; }
    public double getDefaultRelTol() { return defaultRelTol; }
    public Map<String, Double> getOpAbsTol() { return opAbsTol; }
    public boolean isStopAtFirst() { return stopAtFirst; }
    public int getMaxDivergences() { return maxDivergences; }
    public boolean isCaptureInputsForDivergent() { return captureInputsForDivergent; }
    public double getLogitAbsTol() { return logitAbsTol; }
    public int getTopKOverlap() { return topKOverlap; }
}
