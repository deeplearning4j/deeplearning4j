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

import java.util.Locale;
import java.util.Optional;

/**
 * Model-family-owned generation defaults.
 *
 * <p>These defaults apply only when the caller and model artifact do not provide
 * the corresponding generation setting. They are deliberately variant-aware:
 * older LFM2 models do not inherit the LFM2.5 Instruct recommendations.</p>
 */
public final class ModelSamplingDefaults {

    private ModelSamplingDefaults() {
    }

    /**
     * Resolve the published defaults for a model variant.
     *
     * @param family existing SameDiff-LLM model family
     * @param modelIdentity model id and/or artifact model name
     * @return defaults when the exact family variant has published recommendations
     */
    public static Optional<SamplingConfig> forModel(
            ModelFamily family,
            String modelIdentity) {
        if (family == ModelFamily.QWEN35) {
            // Qwen3.5 non-thinking text recommendation:
            // temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
            // presence_penalty=2.0, repetition_penalty=1.0.
            return Optional.of(
                    SamplingConfig.sample(1.0d, 20, 1.0d)
                            .toBuilder()
                            .minP(0.0d)
                            .presencePenalty(2.0d)
                            .repetitionPenalty(1.0d)
                            .build());
        }
        if (family == ModelFamily.LFM2 && isLfm25Instruct(modelIdentity)) {
            // Liquid AI LFM2.5 Instruct recommendation:
            // do_sample=true, temperature=0.1, top_k=50,
            // repetition_penalty=1.05; top_p is unspecified (disabled at 1.0).
            return Optional.of(
                    SamplingConfig.sample(0.1d, 50, 1.0d)
                            .toBuilder()
                            .repetitionPenalty(1.05d)
                            .build());
        }
        return Optional.empty();
    }

    private static boolean isLfm25Instruct(String modelIdentity) {
        if (modelIdentity == null || modelIdentity.isBlank()) {
            return false;
        }
        String normalized = modelIdentity
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]", "");
        return normalized.contains("lfm25") && normalized.contains("instruct");
    }
}
