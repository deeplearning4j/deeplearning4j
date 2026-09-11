/*
 * Copyright (c) Eclipse Deeplearning4j Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.ggml.architecture;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.ggml.convert.ConversionOptions;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Converts the real Gemma GGUF to SDZ with the current builder (including
 * lm_logits_last). Enabled only when the source GGUF path is provided, so
 * normal CI never hits the multi-gigabyte file.
 *
 * Output path defaults to alongside the GGUF; override with
 * -Dgemma4.lastpos.sdz.output to redirect.
 */
class Gemma4LastPositionConversionIT {

    private static final String GGUF_PROPERTY = "gemma4.lastpos.gguf";
    private static final String OUTPUT_PROPERTY = "gemma4.lastpos.sdz.output";

    @Test
    @EnabledIfSystemProperty(named = GGUF_PROPERTY, matches = ".+")
    void convertGemma4WithLastPositionBranch() throws Exception {
        Path gguf = Path.of(System.getProperty(GGUF_PROPERTY));
        assertTrue(Files.isRegularFile(gguf), "GGUF not found: " + gguf);

        Path output = System.getProperty(OUTPUT_PROPERTY) != null
                ? Path.of(System.getProperty(OUTPUT_PROPERTY))
                : gguf.resolveSibling(gguf.getFileName().toString().replaceAll("\\.gguf$", "") + "-lastpos.sdz");

        Files.createDirectories(output.getParent());
        GGMLToSameDiffConverter converter =
                new GGMLToSameDiffConverter(ConversionOptions.forInference());
        converter.convertToSDZ(gguf.toFile(), output.toFile());
        assertTrue(Files.isRegularFile(output) && Files.size(output) > 0,
                "Conversion produced no output: " + output);

        try (SameDiff restored = SDZSerializer.load(output.toFile(), false)) {
            assertTrue(restored.hasVariable("lm_logits_last"),
                    "Restored graph lacks lm_logits_last — builder change missing?");
            assertTrue(restored.hasVariable("actual_sequence_length"),
                    "Restored graph lacks actual_sequence_length");
            assertTrue(restored.outputs().contains("lm_logits_last"),
                    "lm_logits_last not in declared outputs");
        }
    }
}
