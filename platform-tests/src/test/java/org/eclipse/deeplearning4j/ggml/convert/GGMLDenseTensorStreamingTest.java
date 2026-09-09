/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.ggml.convert;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.serde.ModelLoadingContext;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exercises the production GGUF loader with real, locally generated files. These tests
 * bound source reads and staging buffers, not RSS or allocation lifetime. In particular,
 * failure propagation is not evidence that partially allocated native arrays were freed.
 */
class GGMLDenseTensorStreamingTest {
    private static final int MAX_READ_BYTES = 4 * 1024 * 1024;
    private static final String TENSOR_NAME = "dense_probe.weight";

    @TempDir
    Path tempDir;

    @Test
    void largeF16StreamsToFloatWithOddTail() throws Exception {
        assertLargeStreaming(GGMLDataType.GGML_TYPE_F16, DataType.FLOAT);
    }

    @Test
    void largeBFloat16StreamsWithoutChangingBits() throws Exception {
        assertLargeStreaming(GGMLDataType.GGML_TYPE_BF16, DataType.BFLOAT16);
    }

    @Test
    void largeF32StreamsToHalfWithOddTail() throws Exception {
        assertLargeStreaming(GGMLDataType.GGML_TYPE_F32, DataType.HALF);
    }

    @Test
    void largeF64StreamsWithoutFloatNarrowing() throws Exception {
        assertLargeStreaming(GGMLDataType.GGML_TYPE_F64, DataType.DOUBLE);
    }

    @Test
    void floatingTargetsMatchCanonicalCastsIncludingSpecialValues() throws Exception {
        float[] values = {0.0f, -0.0f, 1.003f, -2.011f, 0.33325f, 17.03125f,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN,
                0.00006103515625f, 65504.0f, -0.00006103515625f};
        ByteBuffer bytes = littleEndian(values.length * Float.BYTES);
        for (float value : values) {
            bytes.putFloat(value);
        }
        File file = fixture(GGMLDataType.GGML_TYPE_F32, new long[]{3, 4}, bytes.array());
        for (DataType target : new DataType[]{DataType.FLOAT, DataType.HALF,
                DataType.BFLOAT16, DataType.DOUBLE}) {
            // The reference is the existing ND4J cast contract, not the converter's decoder.
            // Exact post-cast equality deliberately avoids a tolerance that hides rounding errors.
            double[] expected = canonicalCast(values, target);
            try (RecordingReader reader = new RecordingReader(file, false);
                 INDArray output = loadOne(reader, target)) {
                assertLayout(output, target, new long[]{4, 3});
                assertCoverage(reader, values.length * Float.BYTES, Float.BYTES, false);
                double[] actual = floatingValues(output);
                assertEquals(expected.length, actual.length);
                for (int i = 0; i < expected.length; i++) {
                    assertExactValue(expected[i], actual[i], i, target.toString());
                }
            }
        }
    }

    @Test
    void singletonAndOddVectorKeepExactRankAndOwnedStorage() throws Exception {
        for (int length : new int[]{1, 7}) {
            ByteBuffer bytes = littleEndian(length * Short.BYTES);
            for (int i = 0; i < length; i++) {
                bytes.putShort((short) (0x3400 + i));
            }
            File file = fixture(GGMLDataType.GGML_TYPE_F16, new long[]{length}, bytes.array());
            try (RecordingReader reader = new RecordingReader(file, false);
                 INDArray output = loadOne(reader, DataType.FLOAT)) {
                assertLayout(output, DataType.FLOAT, new long[]{length});
                assertCoverage(reader, length * Short.BYTES, Short.BYTES, false);
                double[] actual = floatingValues(output);
                for (int i = 0; i < length; i++) {
                    assertExactValue(Math.scalb(1024.0 + i, -12), actual[i], i, "F16 vector");
                }
            }
        }
    }

    @Test
    void integerPayloadsRetainPrecisionRegardlessOfFloatingTarget() throws Exception {
        // These exceed FLOAT's and DOUBLE's exact-integer ranges, respectively.
        long[][] cases = {
                {Integer.MIN_VALUE, -16_777_217L, 0, 16_777_217L, Integer.MAX_VALUE},
                {Long.MIN_VALUE, -9_007_199_254_740_993L, 0,
                        9_007_199_254_740_993L, Long.MAX_VALUE}
        };
        GGMLDataType[] sources = {GGMLDataType.GGML_TYPE_I32, GGMLDataType.GGML_TYPE_I64};
        for (int c = 0; c < sources.length; c++) {
            GGMLDataType source = sources[c];
            ByteBuffer bytes = littleEndian(cases[c].length * source.getNd4jType().width());
            for (long value : cases[c]) {
                if (source == GGMLDataType.GGML_TYPE_I32) {
                    bytes.putInt((int) value);
                } else {
                    bytes.putLong(value);
                }
            }
            File file = fixture(source, new long[]{cases[c].length}, bytes.array());
            // Integer loading may continue to use the existing whole-tensor reader path.
            try (GGUFReader reader = new GGUFReader(file);
                 INDArray output = loadOne(reader, DataType.HALF)) {
                assertLayout(output, source.getNd4jType(), new long[]{cases[c].length});
                assertArrayEquals(cases[c], output.data().asLong(), source.toString());
            }
        }
    }

    @Test
    void laterRangeIOExceptionFailsClosedWithTensorNameAndOriginalCause() throws Exception {
        int length = MAX_READ_BYTES / Float.BYTES + 3;
        File file = fixture(GGMLDataType.GGML_TYPE_F32, new long[]{length},
                densePayload(GGMLDataType.GGML_TYPE_F32, length));
        try (RecordingReader reader = new RecordingReader(file, true)) {
            IOException failure = assertThrows(IOException.class, () -> {
                // Also close an unexpected successful result before assertThrows fails.
                try (INDArray ignored = loadOne(reader, DataType.FLOAT)) {
                    fail("A partially loaded tensor must not be returned");
                }
            });
            assertTrue(failure.getMessage().contains(TENSOR_NAME), failure.toString());
            Throwable cause = failure;
            while (cause != null && cause != reader.injectedFailure) {
                cause = cause.getCause();
            }
            assertSame(reader.injectedFailure, cause, "Original source IOException was lost");
            assertEquals(2, reader.reads.size(), "Must stop at the failing second range");
            assertTrue(reader.completedBytes > 0, "Failure must follow a successful read");
            assertEquals(reader.completedBytes, reader.reads.get(1).offset);
        }
    }

    private void assertLargeStreaming(GGMLDataType source, DataType target) throws Exception {
        int width = source.getNd4jType().width();
        // Every payload exceeds the byte ceiling, including 16-bit sources. At most
        // about two million elements / four MiB of encoded data; no downloaded model.
        int length = MAX_READ_BYTES / width + 3;
        File file = fixture(source, new long[]{length, 1}, densePayload(source, length));
        try (RecordingReader reader = new RecordingReader(file, false);
             INDArray output = loadOne(reader, target)) {
            assertLayout(output, target, new long[]{1, length});
            assertCoverage(reader, (long) length * width, width, true);
            // One bulk host read, never a per-element synchronized INDArray accessor.
            double[] actual = floatingValues(output);
            assertEquals(length, actual.length);
            String context = source + " -> " + target;
            for (int i = 0; i < length; i++) {
                assertExactValue(denseValue(source, i), actual[i], i, context);
            }
        }
    }

    private File fixture(GGMLDataType source, long[] shape, byte[] payload) throws IOException {
        File file = tempDir.resolve("dense-fixture.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(file, 3)) {
            writer.addMetadataString("general.architecture", "generic");
            // Deliberately put the selected tensor after a non-alignment-sized payload:
            // confusing tensor-relative and file-relative offsets cannot pass numerics.
            writer.registerTensor("prefix", new long[]{17}, GGMLDataType.GGML_TYPE_F32);
            writer.registerTensor(TENSOR_NAME, shape, source);
            writer.writeHeader();
            writer.writeTensorData("prefix", new byte[17 * Float.BYTES]);
            writer.writeTensorData(TENSOR_NAME, payload);
            writer.finalizeFile();
        }
        return file;
    }

    @SuppressWarnings("unchecked")
    private static INDArray loadOne(GGUFReader reader, DataType target) throws Exception {
        GGMLTensorInfo info = reader.getMetadata().getTensors().stream()
                .filter(t -> TENSOR_NAME.equals(t.getName())).findFirst().orElseThrow();
        assertTrue(info.getDataOffset() > 0, "Fixture must exercise a nonzero tensor offset");
        ConversionOptions options = ConversionOptions.builder()
                .architectureOverride("generic")
                .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32)
                .targetDataType(target).build();
        Method method = GGMLToSameDiffConverter.class.getDeclaredMethod("loadWeightsFromGGUF",
                GGUFReader.class, List.class, ModelLoadingContext.class);
        method.setAccessible(true);
        Map<String, INDArray> weights;
        try {
            weights = (Map<String, INDArray>) method.invoke(new GGMLToSameDiffConverter(options),
                    reader, Collections.singletonList(info), null);
        } catch (InvocationTargetException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof Exception) {
                throw (Exception) cause;
            }
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            throw failure;
        }
        if (weights.size() != 1 || weights.get(TENSOR_NAME) == null) {
            for (INDArray value : weights.values()) {
                if (value != null) {
                    value.close();
                }
            }
            fail("Expected exactly the requested dense tensor, got " + weights.keySet());
        }
        return weights.get(TENSOR_NAME);
    }

    private static void assertLayout(INDArray output, DataType type, long[] shape) {
        assertEquals(type, output.dataType());
        assertArrayEquals(shape, output.shape());
        assertEquals('c', output.ordering());
        assertEquals(0L, output.offset());
        assertFalse(output.isView(), "The loaded tensor must own its storage");
        long stride = 1;
        long[] expectedStrides = new long[shape.length];
        for (int i = shape.length - 1; i >= 0; i--) {
            expectedStrides[i] = stride;
            stride *= shape[i];
        }
        // Singleton strides are immaterial; ND4J may canonicalize them to one.
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] > 1) assertEquals(expectedStrides[i], output.stride(i), "Stride at axis " + i);
        }
        assertEquals(stride, output.length());
    }

    private static void assertCoverage(RecordingReader reader, long totalBytes, int width,
                                       boolean requireTail) {
        assertFalse(reader.reads.isEmpty(), "Dense tensors must use range reads");
        long next = 0;
        for (ReadRequest read : reader.reads) {
            assertEquals(next, read.offset, "Ranges must have neither gaps nor overlaps");
            assertEquals(0L, read.offset % width, "Unaligned source offset");
            assertEquals(0, read.length % width, "Partial source element");
            next += read.length;
        }
        assertEquals(totalBytes, next, "Entire payload must be read exactly once");
        assertEquals(totalBytes, reader.completedBytes);
        if (requireTail) {
            assertTrue(reader.reads.size() > 1, "Large fixture must cross a chunk boundary");
            int first = reader.reads.get(0).length;
            int last = reader.reads.get(reader.reads.size() - 1).length;
            assertTrue(last < first, "Expected a full chunk followed by a shorter tail");
            assertEquals(1, (last / width) % 2, "Tail must contain an odd number of elements");
        }
    }

    private static double[] canonicalCast(float[] values, DataType target) {
        try (INDArray source = Nd4j.createFromArray(values)) {
            if (target == DataType.FLOAT) {
                return floatingValues(source);
            }
            try (INDArray cast = source.castTo(target)) {
                return floatingValues(cast);
            }
        }
    }

    private static double[] floatingValues(INDArray array) {
        if (array.dataType() == DataType.DOUBLE) {
            // Read stored doubles directly: CudaDoubleDataBuffer.asDouble() currently
            // converts via FLOAT, which would corrupt the precision oracle itself.
            double[] values = new double[Math.toIntExact(array.length())];
            array.data().asNioDouble().get(values);
            return values;
        }
        return array.data().asDouble();
    }

    private static void assertExactValue(double expected, double actual, int index, String context) {
        // Canonicalize NaN payloads but distinguish +0 from -0 and signed infinities.
        if (Double.doubleToLongBits(expected) != Double.doubleToLongBits(actual)) {
            fail(context + " value " + index + ": expected " + expected + ", got " + actual);
        }
    }

    private static ByteBuffer littleEndian(int bytes) {
        return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static double denseValue(GGMLDataType source, int index) {
        switch (index % 521) {
            case 0: return -0.0;
            case 1: return Double.POSITIVE_INFINITY;
            case 2: return Double.NEGATIVE_INFINITY;
            case 3: return Double.NaN;
            default:
                switch (source) {
                    case GGML_TYPE_F16: return Math.scalb(1024.0 + index % 509, -12);
                    case GGML_TYPE_BF16: return (128.0 + index % 127) / 512.0;
                    case GGML_TYPE_F64: return (index % 509 - 254) / 16.0 + Math.scalb(1.0, -40);
                    default: return (index % 509 - 254) / 16.0;
                }
        }
    }

    private static byte[] densePayload(GGMLDataType source, int length) {
        ByteBuffer bytes = littleEndian(length * source.getNd4jType().width());
        int[] halfSpecial = {0x8000, 0x7c00, 0xfc00, 0x7e00};
        for (int i = 0; i < length; i++) {
            double value = denseValue(source, i);
            switch (source) {
                case GGML_TYPE_F16:
                    bytes.putShort((short) (i % 521 < 4 ? halfSpecial[i % 521] : 0x3400 + i % 509));
                    break;
                case GGML_TYPE_BF16:
                    bytes.putShort((short) (Float.floatToRawIntBits((float) value) >>> 16));
                    break;
                case GGML_TYPE_F32:
                    bytes.putFloat((float) value);
                    break;
                case GGML_TYPE_F64:
                    bytes.putDouble(value);
                    break;
                default:
                    throw new IllegalArgumentException("Not a floating source: " + source);
            }
        }
        return bytes.array();
    }

    private static final class ReadRequest {
        final long offset;
        final int length;

        ReadRequest(long offset, int length) {
            this.offset = offset;
            this.length = length;
        }
    }

    private static final class RecordingReader extends GGUFReader {
        final List<ReadRequest> reads = new ArrayList<>();
        final IOException injectedFailure = new IOException("injected later dense range failure");
        final boolean failSecondRead;
        long completedBytes;

        RecordingReader(File file, boolean failSecondRead) throws IOException {
            super(file);
            this.failSecondRead = failSecondRead;
        }

        @Override
        public byte[] readTensorData(GGMLTensorInfo info) {
            throw new AssertionError("Whole-tensor byte[] read forbidden for " + info.getName());
        }

        @Override
        public ByteBuffer readTensorDataDirect(GGMLTensorInfo info) {
            throw new AssertionError("Whole-tensor direct read forbidden for " + info.getName());
        }

        @Override
        public void readTensorDataRange(GGMLTensorInfo info, long offset, byte[] destination,
                                        int destinationOffset, int length) throws IOException {
            record(info, offset, length, destination.length);
            super.readTensorDataRange(info, offset, destination, destinationOffset, length);
            completedBytes += length;
        }

        @Override
        public void readTensorDataRange(GGMLTensorInfo info, long offset, ByteBuffer destination)
                throws IOException {
            int length = destination.remaining();
            record(info, offset, length, destination.capacity());
            super.readTensorDataRange(info, offset, destination);
            completedBytes += length;
        }

        private void record(GGMLTensorInfo info, long offset, int length, int capacity) throws IOException {
            assertEquals(TENSOR_NAME, info.getName());
            assertTrue(length > 0 && length <= MAX_READ_BYTES,
                    "Dense read exceeds the four MiB bound: " + length);
            assertTrue(capacity <= MAX_READ_BYTES,
                    "Range API must not disguise a whole-tensor staging buffer: " + capacity);
            reads.add(new ReadRequest(offset, length));
            if (failSecondRead && reads.size() == 2) {
                throw injectedFailure;
            }
        }
    }
}
