/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.audio.synthesis;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class PcmWavFileWriterTest {

    @TempDir
    Path temporary;

    @Test
    void writesCompletedPcmWavWithExactHeaderAndSamples() throws Exception {
        PcmWavFileWriter writer = new PcmWavFileWriter(16_000, 16);
        Path output = writer.write(
                Nd4j.createFromArray(-1.0f, -0.5f, 0.0f, 0.5f, 1.0f),
                temporary, "result.wav");

        byte[] bytes = Files.readAllBytes(output);
        assertEquals(44 + 10, bytes.length);
        assertEquals("RIFF", new String(bytes, 0, 4, java.nio.charset.StandardCharsets.US_ASCII));
        assertEquals("WAVE", new String(bytes, 8, 4, java.nio.charset.StandardCharsets.US_ASCII));
        assertEquals("data", new String(bytes, 36, 4, java.nio.charset.StandardCharsets.US_ASCII));

        ByteBuffer littleEndian = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(16_000, littleEndian.getInt(24));
        assertEquals(10, littleEndian.getInt(40));
        assertEquals(Short.MIN_VALUE, littleEndian.getShort(44));
        assertEquals((short) -16383, littleEndian.getShort(46));
        assertEquals((short) 0, littleEndian.getShort(48));
        assertEquals((short) 16384, littleEndian.getShort(50));
        assertEquals(Short.MAX_VALUE, littleEndian.getShort(52));
    }

    @Test
    void streamsLargeWaveformWithoutConstructingWholeFileBytes() throws Exception {
        int samples = 1_000_000;
        PcmWavFileWriter writer = new PcmWavFileWriter(24_000, samples);
        Path output = writer.write(Nd4j.zeros(samples), temporary, "large.wav");

        assertEquals(44L + samples * 2L, Files.size(output));
    }

    @Test
    void runsReusableSameDiffTextToWaveformGeneratorEndToEnd() throws Exception {
        Path modelFile = temporary.resolve("waveform.sdz");
        try (SameDiff sameDiff = SameDiff.create()) {
            SDVariable tokens = sameDiff.placeHolder("tokens", DataType.INT32, -1, -1);
            tokens.castTo(DataType.FLOAT)
                    .div(255.0d)
                    .mul(2.0d)
                    .sub(1.0d)
                    .rename("samples");
            sameDiff.save(modelFile.toFile(), true);
        }

        try (SameDiffWaveformGenerator generator = SameDiffWaveformGenerator.builder()
                .modelFile(modelFile)
                .tokenizerType(SameDiffWaveformGenerator.TokenizerType.UTF8_BYTES)
                .tokenIdsInput("tokens")
                .waveformOutput("samples")
                .tokenDataType(DataType.INT32)
                .sampleRateHz(16_000)
                .maxInputTokens(32)
                .maxOutputSamples(32)
                .voice("standard")
                .language("en")
                .modelId("tiny-waveform")
                .modelVersion("v1")
                .configurationVersion("test-config-v1")
                .defaultConfidence(0.75d)
                .build()) {
            GeneratedAudioFile result = generator.generate(
                    new AudioSynthesisRequest(UUID.randomUUID(), "Hi",
                            "standard", "en", Map.of()),
                    temporary);

            assertTrue(Files.isRegularFile(result.getCompletedFile()));
            assertEquals(48L, Files.size(result.getCompletedFile()));
            assertEquals("audio/wav", result.getMediaType());
            assertEquals("tiny-waveform", result.getModelId());
            assertEquals(0.75d, result.getConfidence(), 0.0d);
            assertEquals(2L, result.getConfigurationEvidence().get("outputSamples"));
        }
    }

    @Test
    void rejectsInvalidSamplesShapesLimitsAndFileNames() {
        PcmWavFileWriter writer = new PcmWavFileWriter(16_000, 4);

        assertThrows(IllegalArgumentException.class,
                () -> writer.write(Nd4j.createFromArray(0.0f, 1.01f),
                        temporary, "range.wav"));
        assertThrows(IllegalArgumentException.class,
                () -> writer.write(Nd4j.createFromArray(Float.NaN),
                        temporary, "nan.wav"));
        assertThrows(IllegalArgumentException.class,
                () -> writer.write(Nd4j.zeros(2, 2),
                        temporary, "shape.wav"));
        assertThrows(IllegalArgumentException.class,
                () -> writer.write(Nd4j.zeros(5),
                        temporary, "limit.wav"));
        assertThrows(IllegalArgumentException.class,
                () -> writer.write(Nd4j.zeros(1),
                        temporary, "../escape.wav"));
    }
}
