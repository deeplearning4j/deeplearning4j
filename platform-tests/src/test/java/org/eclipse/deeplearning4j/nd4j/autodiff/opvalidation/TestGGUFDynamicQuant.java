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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.quantization.DynamicQuantConfig;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GGUF dynamic quantization configuration and export options.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("GGUF Dynamic Quantization Tests")
public class TestGGUFDynamicQuant {

    @Test
    @DisplayName("DynamicQuantConfig - Standard Defaults")
    public void testDynamicQuantConfigDefaults() {
        DynamicQuantConfig config = DynamicQuantConfig.standard();

        assertEquals(0, config.getTargetModelSizeMB());
        assertEquals(ExportOptions.QuantizationType.Q4_K, config.getMinQuantType());
        assertEquals(ExportOptions.QuantizationType.Q6_K, config.getMaxQuantType());
        assertEquals(DynamicQuantConfig.SensitivityMetric.WEIGHT_MAGNITUDE, config.getSensitivityMetric());
        assertEquals(ExportOptions.QuantizationType.F16, config.getEmbeddingPrecision());
        assertEquals(ExportOptions.QuantizationType.Q6_K, config.getAttentionPrecision());

        // Candidate types should include Q4_K, Q5_K, Q6_K
        assertNotNull(config.getCandidateTypes());
        assertEquals(3, config.getCandidateTypes().size());
        assertTrue(config.getCandidateTypes().contains(ExportOptions.QuantizationType.Q4_K));
        assertTrue(config.getCandidateTypes().contains(ExportOptions.QuantizationType.Q5_K));
        assertTrue(config.getCandidateTypes().contains(ExportOptions.QuantizationType.Q6_K));
    }

    @Test
    @DisplayName("DynamicQuantConfig - Unsloth Dynamic 2")
    public void testDynamicQuantConfigUnsloth() {
        DynamicQuantConfig config = DynamicQuantConfig.unslothDynamic2();

        assertEquals(0, config.getTargetModelSizeMB());
        assertEquals(ExportOptions.QuantizationType.Q4_0, config.getMinQuantType());
        assertEquals(ExportOptions.QuantizationType.Q8_0, config.getMaxQuantType());
        assertEquals(DynamicQuantConfig.SensitivityMetric.WEIGHT_MAGNITUDE, config.getSensitivityMetric());
        assertEquals(ExportOptions.QuantizationType.F16, config.getEmbeddingPrecision());
        assertEquals(ExportOptions.QuantizationType.Q6_K, config.getAttentionPrecision());

        // Candidate types should include Q4_0, Q4_K, Q5_K, Q6_K, Q8_0
        assertNotNull(config.getCandidateTypes());
        assertEquals(5, config.getCandidateTypes().size());
        assertTrue(config.getCandidateTypes().contains(ExportOptions.QuantizationType.Q4_0));
        assertTrue(config.getCandidateTypes().contains(ExportOptions.QuantizationType.Q8_0));
    }

    @Test
    @DisplayName("ExportOptions - With Dynamic Quant")
    public void testExportOptionsWithDynamicQuant() {
        DynamicQuantConfig quantConfig = DynamicQuantConfig.standard();
        ExportOptions options = ExportOptions.dynamicQuant(quantConfig);

        assertNotNull(options.getDynamicQuantConfig());
        assertSame(quantConfig, options.getDynamicQuantConfig());
        assertEquals(ExportOptions.QuantizationType.Q4_K, options.getDynamicQuantConfig().getMinQuantType());
        assertEquals(ExportOptions.QuantizationType.Q6_K, options.getDynamicQuantConfig().getMaxQuantType());
    }
}
