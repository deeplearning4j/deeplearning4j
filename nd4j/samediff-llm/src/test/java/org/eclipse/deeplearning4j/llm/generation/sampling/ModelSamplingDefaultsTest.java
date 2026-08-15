/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation.sampling;

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.ModelFamily;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelSamplingDefaultsTest {

    @Test
    void resolvesOfficialLfm25InstructDefaultsThroughExistingLfm2Family() {
        SamplingConfig sampling = ModelSamplingDefaults.forModel(
                        ModelFamily.LFM2,
                        "LiquidAI/LFM2.5-1.2B-Instruct-GGUF")
                .orElseThrow();

        assertTrue(sampling.isDoSample());
        assertEquals(0.1d, sampling.getTemperature());
        assertEquals(50, sampling.getTopK());
        assertEquals(1.0d, sampling.getTopP());
        assertEquals(1.05d, sampling.getRepetitionPenalty());
    }

    @Test
    void resolvesOfficialQwen35NonThinkingTextDefaults() {
        SamplingConfig sampling = ModelSamplingDefaults.forModel(
                        ModelFamily.QWEN35,
                        "Qwen/Qwen3.5-0.8B")
                .orElseThrow();

        assertTrue(sampling.isDoSample());
        assertEquals(1.0d, sampling.getTemperature());
        assertEquals(20, sampling.getTopK());
        assertEquals(1.0d, sampling.getTopP());
        assertEquals(0.0d, sampling.getMinP());
        assertEquals(2.0d, sampling.getPresencePenalty());
        assertEquals(1.0d, sampling.getRepetitionPenalty());
    }

    @Test
    void resolvesOfficialQwen35ThinkingTextDefaultsSeparately() {
        SamplingConfig sampling = ModelSamplingDefaults.forModel(
                        ModelFamily.QWEN35,
                        "Qwen/Qwen3.5-2B",
                        ModelSamplingDefaults.GenerationMode.THINKING_TEXT)
                .orElseThrow();

        assertTrue(sampling.isDoSample());
        assertEquals(1.0d, sampling.getTemperature());
        assertEquals(20, sampling.getTopK());
        assertEquals(0.95d, sampling.getTopP());
        assertEquals(0.0d, sampling.getMinP());
        assertEquals(1.5d, sampling.getPresencePenalty());
        assertEquals(1.0d, sampling.getRepetitionPenalty());
    }

    @Test
    void acceptsNormalizedArtifactNames() {
        assertTrue(ModelSamplingDefaults.forModel(
                ModelFamily.LFM2,
                "LFM2_5-1_2B-INSTRUCT-Q4_K_M.gguf").isPresent());
    }

    @Test
    void doesNotApplyLfm25InstructDefaultsToOlderLfm2Variants() {
        assertFalse(ModelSamplingDefaults.forModel(
                ModelFamily.LFM2,
                "LFM2-1.2B").isPresent());
    }

    @Test
    void doesNotApplyLfmDefaultsToAnotherFamilyWithSimilarText() {
        assertFalse(ModelSamplingDefaults.forModel(
                ModelFamily.MISTRAL,
                "adapter-for-lfm2.5-instruct").isPresent());
    }
}
