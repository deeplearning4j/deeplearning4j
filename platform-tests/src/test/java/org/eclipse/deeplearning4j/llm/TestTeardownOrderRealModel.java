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

package org.eclipse.deeplearning4j.llm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Full-scale reproducer for the DSP teardown-order crash (hs_err_pid1286526,
 * 2026-07-02): after real generation, freeing the decoder's model arrays BEFORE
 * closing the SameDiff SIGSEGV'd inside
 * {@code NativeDynamicShapePlan::releaseGpuIntermediates} — the slot-release loops
 * walked NDArrays whose DataBuffers Java had just freed, and the resulting heap
 * corruption crashed a local {@code unordered_set} lookup
 * ({@code _M_find_before_node}).
 *
 * <p>The small-graph {@code DspTeardownOrderTest} does NOT reproduce this — the crash
 * needs generation-scale state (~1400 slots, hundreds of freed weight buffers,
 * KV-cache views, zero-copy output references). This test replicates the exact
 * crashed sequence: import the cached Qwen3.5-0.8B GGUF, run a short generation
 * through the production pipeline (which owns the optimizer-produced decoder dup),
 * then apply the historical WRONG teardown order manually.</p>
 *
 * <p>Skipped (assumption) when the cached model is absent. Run:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dbackend.artifactId=nd4j-native \
 *       -Dtest=TestTeardownOrderRealModel 2&gt;&amp;1 | tee /tmp/teardown-real.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.LARGE_RESOURCES)
@DisplayName("DSP teardown order at generation scale (real model)")
public class TestTeardownOrderRealModel {

    private static final File GGUF = new File(System.getProperty("user.home"),
            ".cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf");

    @Test
    @DisplayName("free-before-close after real generation must not crash the JVM")
    public void testFreeBeforeCloseAfterGeneration() throws Exception {
        assumeTrue(GGUF.isFile(), "Cached Qwen3.5-0.8B GGUF not present — skipping");

        SameDiff decoder = GGMLModelImport.importModel(GGUF.getAbsolutePath(),
                ConversionOptions.forInference());
        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(GGUF.getParentFile());

        // graphOptimizerEnabled (default true) makes the pipeline own an optimized
        // dup of the decoder — the exact configuration of the crashed run.
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(2)
                .build());

        GenerationResult result = pipeline.generate("The capital of France is", 2);
        assertNotNull(result.getText(), "generation must produce text");
        log.info("Generated: '{}'", result.getText());

        // The pipeline's decoder is the optimizer dup with the warmed DSP plan.
        SameDiff pipelineDecoder = pipeline.getDecoder();

        // Historical WRONG order, applied manually (the pipeline itself now closes
        // first): free every model array, THEN close. The native teardown must
        // survive walking slots whose weight DataBuffers are already gone.
        assertDoesNotThrow(() -> {
            int freed = SameDiffMemoryUtils.freeModelArrays(pipelineDecoder);
            log.info("Freed {} model arrays BEFORE close (historical crash order)", freed);
            pipelineDecoder.close();
        }, "free-before-close at generation scale must be survivable");

        // Normal pipeline close afterwards must also be a safe no-op pass.
        assertDoesNotThrow(pipeline::close, "pipeline.close() after manual teardown must be safe");
        tokenizer.close();
    }
}
