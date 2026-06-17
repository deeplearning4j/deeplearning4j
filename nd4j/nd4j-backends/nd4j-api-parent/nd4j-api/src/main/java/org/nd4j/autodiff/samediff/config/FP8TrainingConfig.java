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

package org.nd4j.autodiff.samediff.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Configuration for FP8 mixed precision training.
 *
 * FP8 training uses 8-bit floating point for forward and backward passes:
 * - E4M3 (4-bit exponent, 3-bit mantissa) for forward activations — higher precision
 * - E5M2 (5-bit exponent, 2-bit mantissa) for gradients — wider dynamic range
 *
 * Per-tensor dynamic scaling tracks the amax (absolute maximum) history
 * to compute optimal scale factors that prevent overflow/underflow.
 *
 * Requires GPU compute capability >= 8.9 (Ada Lovelace / Hopper).
 *
 * Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FP8TrainingConfig implements Serializable {

    /**
     * Use E4M3 format for forward pass activations (higher precision, narrower range).
     * Default: true
     */
    @Builder.Default
    private boolean useE4M3ForForward = true;

    /**
     * Use per-tensor scaling (each tensor gets its own scale factor).
     * When false, uses per-layer scaling (one scale per op).
     * Default: true
     */
    @Builder.Default
    private boolean perTensorScaling = true;

    /**
     * Length of the amax history window for computing scale factors.
     * The scale is computed from the max of the last N amax values.
     * Default: 16
     */
    @Builder.Default
    private int amaxHistoryLength = 16;

    /**
     * Operations eligible for FP8 computation.
     * Typically only matmul/linear operations benefit from FP8.
     * Default: {"matmul", "linear", "dense"}
     */
    @Builder.Default
    private Set<String> fp8EligibleOps = new HashSet<>(Arrays.asList("matmul", "linear", "dense"));

    /**
     * Operations that should never use FP8 (normalization, softmax, etc.
     * where precision is critical).
     * Default: {"layernorm", "rmsnorm", "softmax", "attention"}
     */
    @Builder.Default
    private Set<String> excludeFromFP8 = new HashSet<>(Arrays.asList(
            "layernorm", "rmsnorm", "softmax", "attention", "gelu", "silu"
    ));

    /**
     * Validate the configuration.
     */
    public void validate() {
        if (amaxHistoryLength < 1) {
            throw new IllegalStateException("amaxHistoryLength must be >= 1, got: " + amaxHistoryLength);
        }
    }

    /**
     * Check if a given op type is eligible for FP8 computation.
     */
    public boolean isEligibleForFP8(String opType) {
        if (excludeFromFP8 != null && excludeFromFP8.contains(opType)) {
            return false;
        }
        return fp8EligibleOps != null && fp8EligibleOps.contains(opType);
    }
}
