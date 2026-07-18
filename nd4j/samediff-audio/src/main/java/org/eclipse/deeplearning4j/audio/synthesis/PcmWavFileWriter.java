/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.audio.synthesis;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Streams a normalized mono waveform to a completed 16-bit little-endian PCM
 * WAV file.
 *
 * <p>The writer never materializes the encoded audio as a whole-file byte
 * array. It rejects non-finite and out-of-range samples instead of silently
 * normalizing or transcoding model output.</p>
 */
public final class PcmWavFileWriter {

    public static final String MEDIA_TYPE = "audio/wav";
    public static final String SAMPLE_FORMAT = "pcm_s16le";

    private static final int HEADER_BYTES = 44;
    private static final int BYTES_PER_SAMPLE = 2;
    private static final int STREAM_BUFFER_BYTES = 64 * 1024;
    private static final long MAX_UNSIGNED_INT = 0xffff_ffffL;

    private final int sampleRateHz;
    private final long maxSamples;

    public PcmWavFileWriter(int sampleRateHz, long maxSamples) {
        if (sampleRateHz <= 0) {
            throw new IllegalArgumentException("sampleRateHz must be positive");
        }
        if (maxSamples <= 0) {
            throw new IllegalArgumentException("maxSamples must be positive");
        }
        this.sampleRateHz = sampleRateHz;
        this.maxSamples = maxSamples;
    }

    /**
     * Write one completed file inside {@code outputDirectory}.
     *
     * @param waveform normalized mono samples, rank 1 or rank 2 with one singleton dimension
     * @param outputDirectory serving-owned output directory
     * @param fileName fixed serving-generated file name, not user input
     * @return absolute normalized path to the completed file
     */
    public Path write(INDArray waveform, Path outputDirectory, String fileName) throws IOException {
        validateWaveformShape(waveform);
        Path directory = requireOutputDirectory(outputDirectory);
        String safeName = requireFileName(fileName);
        Path completed = directory.resolve(safeName).normalize();
        if (!completed.startsWith(directory)) {
            throw new IllegalArgumentException("fileName escaped outputDirectory");
        }

        long sampleCount = waveform.length();
        long dataLength = Math.multiplyExact(sampleCount, BYTES_PER_SAMPLE);
        if (dataLength > MAX_UNSIGNED_INT || 36L + dataLength > MAX_UNSIGNED_INT) {
            throw new IllegalArgumentException("waveform is too large for a RIFF/WAV file");
        }

        Path temporary = Files.createTempFile(directory, ".pcm-wav-", ".tmp");
        boolean moved = false;
        try {
            try (FileChannel channel = FileChannel.open(temporary,
                    StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                writeHeader(channel, dataLength);
                writeSamples(channel, waveform);
                channel.force(true);
            }
            moveCompleted(temporary, completed);
            moved = true;
            return completed;
        } finally {
            if (!moved) {
                Files.deleteIfExists(temporary);
            }
        }
    }

    private void validateWaveformShape(INDArray waveform) {
        Objects.requireNonNull(waveform, "waveform");
        if (!waveform.dataType().isFPType()) {
            throw new IllegalArgumentException("waveform must use a floating-point data type");
        }
        if (waveform.isEmpty() || waveform.length() <= 0) {
            throw new IllegalArgumentException("waveform must contain at least one sample");
        }
        if (waveform.length() > maxSamples) {
            throw new IllegalArgumentException("waveform exceeds the configured sample limit");
        }
        if (waveform.rank() == 1) {
            return;
        }
        if (waveform.rank() == 2 && (waveform.size(0) == 1 || waveform.size(1) == 1)) {
            return;
        }
        throw new IllegalArgumentException(
                "waveform must be rank 1 or rank 2 with one singleton dimension");
    }

    private Path requireOutputDirectory(Path outputDirectory) throws IOException {
        Path normalized = Objects.requireNonNull(outputDirectory, "outputDirectory")
                .toAbsolutePath().normalize();
        if (!Files.isDirectory(normalized, LinkOption.NOFOLLOW_LINKS)
                || Files.isSymbolicLink(normalized)) {
            throw new IllegalArgumentException("outputDirectory must be a regular directory");
        }
        return normalized.toRealPath(LinkOption.NOFOLLOW_LINKS);
    }

    private String requireFileName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            throw new IllegalArgumentException("fileName must not be blank");
        }
        Path name = Path.of(fileName);
        if (name.isAbsolute() || name.getNameCount() != 1 || !name.toString().equals(fileName)) {
            throw new IllegalArgumentException("fileName must be one local path segment");
        }
        return fileName;
    }

    private void writeHeader(FileChannel channel, long dataLength) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        putAscii(header, "RIFF");
        header.putInt((int) (36L + dataLength));
        putAscii(header, "WAVE");
        putAscii(header, "fmt ");
        header.putInt(16);
        header.putShort((short) 1);
        header.putShort((short) 1);
        header.putInt(sampleRateHz);
        header.putInt(Math.multiplyExact(sampleRateHz, BYTES_PER_SAMPLE));
        header.putShort((short) BYTES_PER_SAMPLE);
        header.putShort((short) 16);
        putAscii(header, "data");
        header.putInt((int) dataLength);
        header.flip();
        writeFully(channel, header);
    }

    private void writeSamples(FileChannel channel, INDArray waveform) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocateDirect(STREAM_BUFFER_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (long index = 0; index < waveform.length(); index++) {
            double sample = waveform.getDouble(index);
            if (!Double.isFinite(sample) || sample < -1.0d || sample > 1.0d) {
                throw new IllegalArgumentException(
                        "waveform sample " + index + " must be finite and between -1 and 1");
            }
            if (buffer.remaining() < BYTES_PER_SAMPLE) {
                buffer.flip();
                writeFully(channel, buffer);
                buffer.clear();
            }
            short pcm = sample == -1.0d
                    ? Short.MIN_VALUE
                    : (short) Math.round(sample * Short.MAX_VALUE);
            buffer.putShort(pcm);
        }
        buffer.flip();
        writeFully(channel, buffer);
    }

    private static void writeFully(FileChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }

    private static void putAscii(ByteBuffer buffer, String value) {
        for (int index = 0; index < value.length(); index++) {
            buffer.put((byte) value.charAt(index));
        }
    }

    private static void moveCompleted(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
