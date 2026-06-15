/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.tests;

/**
 * Standard tolerance constants for numeric assertions across the test suite.
 * Use these instead of hardcoding epsilon values in individual tests.
 *
 * Selection guide:
 * - FP64_STRICT: double-precision ops where exact results are expected
 * - FP32_STRICT: single-precision ops, gradient checks, most op validation
 * - FP32_RELAXED: ops with known numerical instability (reductions over large arrays, etc.)
 * - FP16: half-precision ops or ops running on FP16 hardware paths
 * - TF32: TensorFloat-32 paths (CUDA Ampere+ with TF32 enabled)
 * - QUANTIZED: quantized ops (INT8/INT4 inference, GGML dequantization)
 */
public final class TestTolerances {

    private TestTolerances() {}

    /** Double-precision strict: for ops that should be nearly exact in FP64 */
    public static final double FP64_STRICT = 1e-8;

    /** Single-precision strict: standard tolerance for FP32 op validation and gradient checks */
    public static final double FP32_STRICT = 1e-5;

    /** Single-precision relaxed: for ops with accumulation error or numerical instability */
    public static final double FP32_RELAXED = 1e-3;

    /** Half-precision: for FP16 compute paths */
    public static final double FP16 = 1e-2;

    /** TensorFloat-32: CUDA TF32 mode (19-bit mantissa) */
    public static final double TF32 = 5e-3;

    /** Quantized: for INT8/INT4/GGML quantized inference */
    public static final double QUANTIZED = 0.1;
}
