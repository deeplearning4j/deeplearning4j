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

package org.nd4j.ggml.format;

import lombok.Builder;
import lombok.Data;

/**
 * Legacy GGML file header (pre-GGUF format).
 * Supports GGML, GGMF, and GGJT format variants.
 */
@Data
@Builder(toBuilder = true)
public class GGMLHeader {

    /**
     * Magic number identifying the format variant
     */
    private int magic;

    /**
     * Format version
     */
    private int version;

    /**
     * Number of tensors in the file
     */
    private long tensorCount;

    // Model hyperparameters (layout varies by model type)
    private int vocabSize;
    private int embeddingSize;
    private int hiddenSize;
    private int numLayers;
    private int numHeads;
    private int numKVHeads;
    private int contextLength;
    private int fileType;

    /**
     * Determine the legacy format variant
     */
    public LegacyFormat getLegacyFormat() {
        switch (magic) {
            case 0x67676D6C: return LegacyFormat.GGML;
            case 0x67676D66: return LegacyFormat.GGMF;
            case 0x67676A74: return LegacyFormat.GGJT;
            default: return LegacyFormat.UNKNOWN;
        }
    }

    /**
     * Legacy GGML format variants
     */
    public enum LegacyFormat {
        GGML,   // Original format
        GGMF,   // GGML with FP16 support
        GGJT,   // GGML with tensor alignment
        UNKNOWN
    }

    /**
     * Get the GGML file type (indicates quantization)
     */
    public GGMLFileType getGGMLFileType() {
        return GGMLFileType.fromValue(fileType);
    }

    /**
     * GGML file types (quantization levels)
     */
    public enum GGMLFileType {
        ALL_F32(0),
        MOSTLY_F16(1),
        MOSTLY_Q4_0(2),
        MOSTLY_Q4_1(3),
        MOSTLY_Q4_1_SOME_F16(4),
        MOSTLY_Q8_0(7),
        MOSTLY_Q5_0(8),
        MOSTLY_Q5_1(9),
        MOSTLY_Q2_K(10),
        MOSTLY_Q3_K_S(11),
        MOSTLY_Q3_K_M(12),
        MOSTLY_Q3_K_L(13),
        MOSTLY_Q4_K_S(14),
        MOSTLY_Q4_K_M(15),
        MOSTLY_Q5_K_S(16),
        MOSTLY_Q5_K_M(17),
        MOSTLY_Q6_K(18),
        UNKNOWN(-1);

        private final int value;

        GGMLFileType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }

        public static GGMLFileType fromValue(int value) {
            for (GGMLFileType type : values()) {
                if (type.value == value) {
                    return type;
                }
            }
            return UNKNOWN;
        }
    }
}
