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
import org.nd4j.linalg.api.buffer.DataType;
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
    private static final String SKIP_OPTIMIZE_PROPERTY = "gemma4.lastpos.skipOptimize";

    @Test
    @EnabledIfSystemProperty(named = GGUF_PROPERTY, matches = ".+")
    void convertGemma4WithLastPositionBranch() throws Exception {
        Path gguf = Path.of(System.getProperty(GGUF_PROPERTY));
        assertTrue(Files.isRegularFile(gguf), "GGUF not found: " + gguf);

        Path output = System.getProperty(OUTPUT_PROPERTY) != null
                ? Path.of(System.getProperty(OUTPUT_PROPERTY))
                : gguf.resolveSibling(gguf.getFileName().toString().replaceAll("\\.gguf$", "") + "-lastpos.sdz");

        Files.createDirectories(output.getParent());
        // Runtime-quantized (packed) weights keep the Q4_K_M payload packed behind
        // ggml_qmatmul ops: ~4.5 GB instead of ~10.4 GB dequantized FP16. This is
        // the memory profile the e2e Gemma suite passes with under 14 GiB/4 GiB
        // device caps, and it keeps the runtime optimizer's SameDiff.dup affordable.
        GGMLToSameDiffConverter converter =
                new GGMLToSameDiffConverter(ConversionOptions.builder()
                        .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                        .targetDataType(DataType.HALF)
                        .build());
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

        if (Boolean.getBoolean(SKIP_OPTIMIZE_PROPERTY)) {
            // Diagnostic discriminator: save the raw converted graph without the
            // optimizer pass so a runtime failure can be attributed to conversion
            // vs optimization.
            SDZSerializer.save(sd, output.toFile(), false, null);
            System.out.println("Saved raw SDZ, optimizer skipped (" + sd.getOps().size()
                    + " ops) to " + output);
            return;
        }

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs());
        if (optimized != sd) {
            SDZSerializer.save(optimized, output.toFile(), false, null);
            System.out.println("Saved optimized SDZ (" + optimized.getOps().size()
                    + " ops) to " + output);
        } else {
            // Optimizer returned the original graph unchanged (runtime-quantized
            // graphs are eligible only for safe passes). Re-saving is still required
            // here: convertToSDZ output was produced by the raw builder and this
            // load/save round-trip preserves the lastpos structural assertions.
            SDZSerializer.save(sd, output.toFile(), false, null);
            System.out.println("Optimizer returned original graph; re-saved unchanged ("
                    + sd.getOps().size() + " ops) to " + output);
        }
    }
}
