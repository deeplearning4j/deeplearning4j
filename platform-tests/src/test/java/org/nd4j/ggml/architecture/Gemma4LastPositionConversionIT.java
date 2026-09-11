/*
 * Copyright (c) Eclipse Deeplearning4j Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.ggml.architecture;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
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

        // Optimize the graph and save the optimized version so the runtime
        // doesn't need SameDiff.dup() during GraphOptimizer (which doubles
        // model memory and OOMs under device caps).
        SameDiff sd = SDZSerializer.load(output.toFile(), false);
        assertTrue(sd.hasVariable("lm_logits_last"),
                "Restored graph lacks lm_logits_last — builder change missing?");
        assertTrue(sd.hasVariable("actual_sequence_length"),
                "Restored graph lacks actual_sequence_length");
        assertTrue(sd.outputs().contains("lm_logits_last"),
                "lm_logits_last not in declared outputs");

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs());
        SDZSerializer.save(optimized, output.toFile(), false, null);
        System.out.println("Saved optimized SDZ (" + optimized.getOps().size()
                + " ops) to " + output);
    }
}
