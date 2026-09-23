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
 * Converts the Qwen3.5-2B Q4_K_M GGUF to a runtime-quantized HALF SDZ using the
 * same proven profile as {@link Gemma4LastPositionConversionIT}:
 * {@code RUNTIME_QUANTIZED_MATMUL + HALF}. Weights stay packed behind ggml_qmatmul
 * ops, so the loaded artifact no longer carries full-FP32 weight bytes and the
 * runtime optimizer's FP16 pre-cast has nothing to expand.
 *
 * <p>Enabled only when {@code -Dqwen352b.sdz.gguf} is provided so normal CI never
 * hits the multi-gigabyte file. Output to {@code -Dqwen352b.sdz.output} (default:
 * sibling of the GGUF). Replaces the accidentally full-FP32 SDZ that the FP&A
 * capture test was loading (3,760,193,536 bytes of appendable FLOAT data).</p>
 */
class Qwen35LastPositionConversionIT {

    private static final String GGUF_PROPERTY = "qwen352b.sdz.gguf";
    private static final String OUTPUT_PROPERTY = "qwen352b.sdz.output";
    private static final String SKIP_OPTIMIZE_PROPERTY = "qwen352b.sdz.skipOptimize";

    @Test
    @EnabledIfSystemProperty(named = GGUF_PROPERTY, matches = ".+")
    void convertQwen35TwoBToRuntimeQuantizedHalfSdz() throws Exception {
        Path gguf = Path.of(System.getProperty(GGUF_PROPERTY));
        assertTrue(Files.isRegularFile(gguf), "GGUF not found: " + gguf);

        Path output = System.getProperty(OUTPUT_PROPERTY) != null
                ? Path.of(System.getProperty(OUTPUT_PROPERTY))
                : gguf.resolveSibling(gguf.getFileName().toString().replaceAll("\\.gguf$", "") + "-rq.sdz");

        Files.createDirectories(output.getParent());
        GGMLToSameDiffConverter converter =
                new GGMLToSameDiffConverter(ConversionOptions.builder()
                        .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                        .targetDataType(DataType.HALF)
                        .build());
        converter.convertToSDZ(gguf.toFile(), output.toFile());
        assertTrue(Files.isRegularFile(output) && Files.size(output) > 0,
                "Conversion produced no output: " + output);

        SameDiff sd = SDZSerializer.load(output.toFile(), false);
        assertTrue(sd.hasVariable("actual_sequence_length"),
                "Restored graph lacks actual_sequence_length");

        if (Boolean.getBoolean(SKIP_OPTIMIZE_PROPERTY)) {
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
            SDZSerializer.save(sd, output.toFile(), false, null);
            System.out.println("Optimizer returned original graph; re-saved unchanged ("
                    + sd.getOps().size() + " ops) to " + output);
        }
    }
}
