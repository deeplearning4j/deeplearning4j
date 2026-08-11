/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.config;

import java.util.Locale;

/**
 * Storage policy for inference weights.
 *
 * <p>Floating-point policies are dense ND4J data types. INT8 and INT4 are packed
 * quantized-weight policies and therefore require a format-aware importer and
 * quantized matmul kernels; they are not integer casts of dense weights.</p>
 */
public enum ND4JInferenceWeightDataType {
    FLOAT32("fp32"),
    FLOAT16("fp16"),
    BFLOAT16("bf16"),
    FLOAT8_E4M3("fp8"),
    FLOAT8_E5M2("fp8_e5m2"),
    INT8("int8"),
    INT4("int4");

    private final String canonicalName;

    ND4JInferenceWeightDataType(String canonicalName) {
        this.canonicalName = canonicalName;
    }

    public String canonicalName() {
        return canonicalName;
    }

    public boolean isPackedInteger() {
        return this == INT8 || this == INT4;
    }

    /**
     * Resolve the process-wide inference weight policy.
     *
     * <p>The explicit {@code nd4j.optimizer.weightDtype} setting wins. The
     * legacy BF16 and FP16 booleans remain compatible when the new setting is
     * absent. With no configuration, FP16 is the default.</p>
     */
    public static ND4JInferenceWeightDataType resolve() {
        String configured = System.getProperty(ND4JSystemProperties.OPTIMIZER_WEIGHT_DTYPE);
        if (configured != null && !configured.trim().isEmpty()) {
            return fromString(configured);
        }
        if ("true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.OPTIMIZER_BF16))) {
            return BFLOAT16;
        }
        String legacyFp16 = System.getProperty(ND4JSystemProperties.OPTIMIZER_FP16);
        if (legacyFp16 != null && "false".equalsIgnoreCase(legacyFp16.trim())) {
            return FLOAT32;
        }
        return FLOAT16;
    }

    public static ND4JInferenceWeightDataType fromString(String value) {
        if (value == null || value.trim().isEmpty()) {
            return FLOAT16;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT)
                .replace("_", "")
                .replace("-", "")
                .replace(".", "");
        switch (normalized) {
            case "float":
            case "float32":
            case "fp32":
            case "f32":
                return FLOAT32;
            case "half":
            case "float16":
            case "fp16":
            case "f16":
                return FLOAT16;
            case "bfloat16":
            case "bf16":
                return BFLOAT16;
            case "float8":
            case "fp8":
            case "float8e4m3":
            case "fp8e4m3":
            case "e4m3":
                return FLOAT8_E4M3;
            case "float8e5m2":
            case "fp8e5m2":
            case "e5m2":
                return FLOAT8_E5M2;
            case "int8":
            case "i8":
            case "q8":
            case "q80":
                return INT8;
            case "int4":
            case "i4":
            case "q4":
            case "q4k":
                return INT4;
            default:
                throw new IllegalArgumentException("Unsupported inference weight data type '" + value
                        + "'. Supported values: fp32, fp16, bf16, fp8, fp8_e5m2, int8, int4");
        }
    }
}
