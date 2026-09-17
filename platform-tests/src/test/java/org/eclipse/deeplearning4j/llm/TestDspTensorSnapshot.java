/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm;

import org.bytedeco.javacpp.BytePointer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Offline wire-format fixtures, not a substitute for the parent's native capture
 * integration run. No model, download or graph execution. Select the backend via
 * -Dbackend.artifactId when running from platform-tests.
 */
class TestDspTensorSnapshot {
    @TempDir Path directory;

    @Test void explicitRequestIsNotAnOrdinaryCategory() {
        assertEquals(0, DspDiagnostics.ALL & DspDiagnostics.TENSOR_SNAPSHOT);
        assertEquals(0, DspDiagnostics.parseCategories("ALL,KV_CACHE") & DspDiagnostics.TENSOR_SNAPSHOT);
        assertEquals(DspDiagnostics.ALL | DspDiagnostics.TENSOR_SNAPSHOT,
                DspDiagnostics.parseCategories("all, tensor_snapshot"));
    }

    @Test void rawBfloatRoundTripPreservesBitsAndLayouts() throws Exception {
        long[][] shapes = {{2, 3}, {2, 3}, {2, 2}, {}, {0, 3}};
        long[][] strides = {{3, 1}, {1, 2}, {6, 2}, {}, {3, 1}};
        char[] orders = {'c', 'f', 'c', 'c', 'c'};
        int[] counts = {6, 6, 4, 1, 0};
        for (ByteOrder endian : new ByteOrder[]{ByteOrder.LITTLE_ENDIAN, ByteOrder.BIG_ENDIAN}) {
            for (int layout = 0; layout < shapes.length; layout++) {
                ByteBuffer bits = ByteBuffer.allocate(counts[layout] * 2).order(endian);
                short[] values = {(short) 0x8000, (short) 0x7fc1, 1, (short) 0x7f80, (short) 0xff80, (short) 0x3f81};
                for (int i = 0; i < counts[layout]; i++) bits.putShort(values[i]);
                byte[] raw = bits.array();
                Path path = directory.resolve("layout-" + layout + "-" + endian + ".dspt");
                Files.write(path, fixture(DataType.BFLOAT16, shapes[layout], strides[layout], orders[layout], 5, endian, raw));
                DspTensorSnapshot snapshot = DspTensorSnapshot.read(path);
                assertEquals(3, snapshot.callIndex); assertEquals(143, snapshot.sourcePosition);
                assertEquals("executeMtpCuda/pre-executeSteadyState", snapshot.source);
                DspTensorSnapshot.Tensor tensor = snapshot.tensors.get("input");
                assertEquals(5, tensor.originalOffset);
                assertArrayEquals(raw, tensor.raw);
                Map<String, INDArray> first = snapshot.reconstruct(), second = snapshot.reconstruct();
                try {
                    INDArray a = first.get("input"), b = second.get("input");
                    assertEquals(DataType.BFLOAT16, a.dataType());
                    assertArrayEquals(shapes[layout], a.shape());
                    if (raw.length == 0) { assertEquals(0, a.length()); continue; }
                    assertArrayEquals(strides[layout], a.stride());
                    assertNotEquals(a.data().address(), b.data().address());
                    assertArrayEquals(raw, storageBits(a, tensor, endian));
                    assertArrayEquals(raw, storageBits(b, tensor, endian));
                    a.assign(0);
                    assertArrayEquals(raw, storageBits(b, tensor, endian), "Reconstructed tensors must not share storage");
                    tensor.restore(a, endian);
                    assertArrayEquals(raw, storageBits(a, tensor, endian), "Restoring stable feeds must preserve every bit");
                } finally {
                    first.values().forEach(INDArray::close); second.values().forEach(INDArray::close);
                }
            }
        }
    }

    @Test void scalarIntegerAndFloatStorage() throws Exception {
        for (DataType type : new DataType[]{DataType.INT64, DataType.FLOAT, DataType.DOUBLE, DataType.HALF}) {
            byte[] raw = new byte[type.width()]; Arrays.fill(raw, (byte) 0xa5);
            DspTensorSnapshot s = DspTensorSnapshot.decode(fixture(type, new long[0], new long[0], 'c', 0,
                    ByteOrder.LITTLE_ENDIAN, raw));
            Map<String, INDArray> tensors = s.reconstruct();
            try { assertArrayEquals(raw, storageBits(tensors.get("input"), s.tensors.get("input"), s.payloadOrder)); }
            finally { tensors.values().forEach(INDArray::close); }
        }
    }

    @Test void sevenNineAndTenTensorFramesKeepOutputsOutOfFeeds() throws Exception {
        for (int count : new int[]{7, 9, 10}) {
            for (int call = 1; call <= 3; call++) {
                byte[] bytes = mtpFixture(count, call);
                DspTensorSnapshot capture = DspTensorSnapshot.decode(bytes);
                assertEquals(call, capture.callIndex);
                assertEquals(count, capture.tensors.size());
                Map<String, INDArray> feeds = capture.reconstructInputs();
                try {
                    assertEquals(DspTensorSnapshot.MTP_INPUTS, new ArrayList<>(feeds.keySet()));
                    for (String name : feeds.keySet()) {
                        feeds.get(name).assign(0);
                        capture.tensors.get(name).restore(feeds.get(name), capture.payloadOrder);
                        assertArrayEquals(capture.tensors.get(name).raw,
                                storageBits(feeds.get(name), capture.tensors.get(name), capture.payloadOrder));
                    }
                    assertFalse(feeds.containsKey(DspTensorSnapshot.LOGITS));
                    assertFalse(feeds.containsKey(DspTensorSnapshot.HIDDEN));
                    assertFalse(feeds.containsKey(DspTensorSnapshot.DRAFT));
                    if (count >= 9) {
                        DspTensorSnapshot.Tensor logits = capture.tensors.get(DspTensorSnapshot.LOGITS);
                        try (INDArray a = logits.reconstruct(capture.payloadOrder)) {
                            assertEquals(DataType.FLOAT, a.dataType());
                            assertArrayEquals(new long[]{1, 1, 3}, a.shape());
                            assertArrayEquals(new long[]{6, 6, 2}, a.stride());
                            assertArrayEquals(logits.raw, storageBits(a, logits, capture.payloadOrder));
                        }
                        DspTensorSnapshot.Tensor hidden = capture.tensors.get(DspTensorSnapshot.HIDDEN);
                        try (INDArray a = hidden.reconstruct(capture.payloadOrder)) {
                            assertEquals(DataType.BFLOAT16, a.dataType());
                            assertArrayEquals(hidden.raw, storageBits(a, hidden, capture.payloadOrder));
                        }
                    }
                    if (count == 10) {
                        try (INDArray draft = capture.tensors.get(DspTensorSnapshot.DRAFT).reconstruct(capture.payloadOrder)) {
                            assertEquals(DataType.INT64, draft.dataType());
                            assertEquals(0, draft.rank());
                            assertEquals(2, draft.getLong(0));
                        }
                        byte[] corruptOutput = bytes.clone(); corruptOutput[corruptOutput.length - 9] ^= 1;
                        assertThrows(IOException.class, () -> DspTensorSnapshot.decode(corruptOutput));
                    }
                } finally { feeds.values().forEach(INDArray::close); }
            }
        }
    }

    private static byte[] mtpFixture(int count, int call) {
        List<String> names = new ArrayList<>(DspTensorSnapshot.MTP_INPUTS);
        if (count >= 9) { names.add(DspTensorSnapshot.LOGITS); names.add(DspTensorSnapshot.HIDDEN); }
        if (count == 10) names.add(DspTensorSnapshot.DRAFT);
        ByteBuffer out = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN);
        out.put("DSPTS001".getBytes(StandardCharsets.US_ASCII)); out.putInt(call); out.putLong(140 + call);
        string(out, count == 7 ? "executeMtpCuda/pre-executeSteadyState"
                : "executeMtpCuda/pre-executeSteadyState;stream=123;enqueuePhases=" + (count == 9 ? 2 : 3)
                + ";tensors=" + count);
        out.putInt(count); out.put((byte) 1);
        List<byte[]> payloads = new ArrayList<>();
        for (String name : names) {
            boolean logits = name.equals(DspTensorSnapshot.LOGITS), hidden = name.equals(DspTensorSnapshot.HIDDEN);
            DataType dtype = logits ? DataType.FLOAT : hidden ? DataType.BFLOAT16 : DataType.INT64;
            long[] shape = logits ? new long[]{1, 1, 3} : hidden ? new long[]{1, 1, 2} : new long[0];
            long[] strides = logits ? new long[]{6, 6, 2} : hidden ? new long[]{2, 2, 1} : new long[0];
            ByteBuffer raw = ByteBuffer.allocate((logits ? 3 : hidden ? 2 : 1) * dtype.width()).order(ByteOrder.LITTLE_ENDIAN);
            if (logits) raw.putFloat(-1).putFloat(0).putFloat(3);
            else if (hidden) raw.putShort((short) 0x3f81).putShort((short) 0x8000);
            else raw.putLong(name.equals(DspTensorSnapshot.DRAFT) ? 2 : call);
            string(out, name); out.putInt(dtype.toInt()); out.putInt(dtype.width());
            out.putInt(shape.length); out.put((byte) 'c'); out.putLong(logits ? 3 : 0);
            for (int d = 0; d < shape.length; d++) { out.putLong(shape[d]); out.putLong(strides[d]); }
            out.putLong(raw.capacity()); out.putLong(DspTensorSnapshot.checksum(raw.array()));
            payloads.add(raw.array());
        }
        for (byte[] raw : payloads) out.put(raw);
        out.put("DSPTDONE".getBytes(StandardCharsets.US_ASCII));
        return Arrays.copyOf(out.array(), out.position());
    }

    @Test void rejectsIncompleteCorruptAndOversizedPayloads() throws Exception {
        byte[] valid = fixture(DataType.BFLOAT16, new long[]{2}, new long[]{1}, 'c', 0,
                ByteOrder.LITTLE_ENDIAN, new byte[]{1, 2, 3, 4});
        assertThrows(IOException.class, () -> DspTensorSnapshot.decode(Arrays.copyOf(valid, valid.length - 1)));
        byte[] corrupt = valid.clone(); corrupt[corrupt.length - 9] ^= 1;
        assertThrows(IOException.class, () -> DspTensorSnapshot.decode(corrupt));
        byte[] version = valid.clone(); version[7] = '9';
        assertThrows(IOException.class, () -> DspTensorSnapshot.decode(version));
        assertThrows(IOException.class, () -> DspTensorSnapshot.decode(fixture(DataType.DOUBLE,
                new long[]{Long.MAX_VALUE, 2}, new long[]{2, 1}, 'c', 0, ByteOrder.LITTLE_ENDIAN, new byte[0])));
        assertThrows(IOException.class, () -> DspTensorSnapshot.decode(fixture(DataType.BFLOAT16,
                new long[]{DspTensorSnapshot.BUDGET}, new long[]{1}, 'c', 0, ByteOrder.LITTLE_ENDIAN, new byte[0])));
        assertEquals(0xcbf29ce484222325L, DspTensorSnapshot.checksum(new byte[0]));
        assertEquals(0xa430d84680aabd0bL, DspTensorSnapshot.checksum("hello".getBytes(StandardCharsets.US_ASCII)));
    }

    private static byte[] storageBits(INDArray a, DspTensorSnapshot.Tensor t, ByteOrder endian) {
        BytePointer p = new BytePointer(a.data().pointer()).capacity(a.data().length() * t.width);
        byte[] result = new byte[t.raw.length];
        for (int i = 0; i < result.length / t.width; i++) {
            long offset = (a.offset() + t.logicalOffset(i)) * t.width;
            for (int b = 0; b < t.width; b++) {
                int nativeByte = endian == ByteOrder.nativeOrder() ? b : t.width - 1 - b;
                result[i * t.width + b] = p.get(offset + nativeByte);
            }
        }
        return result;
    }

    // Independent encoder for the documented native wire layout. Native capture
    // writes exactly this header, followed by all tensors' raw payloads and footer.
    private static byte[] fixture(DataType dtype, long[] shape, long[] strides, char order,
                                  long offset, ByteOrder endian, byte[] raw) {
        ByteBuffer out = ByteBuffer.allocate(4096 + raw.length).order(ByteOrder.LITTLE_ENDIAN);
        out.put("DSPTS001".getBytes(StandardCharsets.US_ASCII)); out.putInt(3); out.putLong(143);
        string(out, "executeMtpCuda/pre-executeSteadyState"); out.putInt(1);
        out.put((byte) (endian == ByteOrder.LITTLE_ENDIAN ? 1 : 0));
        string(out, "input"); out.putInt(dtype.toInt()); out.putInt(dtype.width());
        out.putInt(shape.length); out.put((byte) order); out.putLong(offset);
        for (int i = 0; i < shape.length; i++) { out.putLong(shape[i]); out.putLong(strides[i]); }
        out.putLong(raw.length); out.putLong(DspTensorSnapshot.checksum(raw)); out.put(raw);
        out.put("DSPTDONE".getBytes(StandardCharsets.US_ASCII));
        return Arrays.copyOf(out.array(), out.position());
    }
    private static void string(ByteBuffer out, String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8); out.putInt(bytes.length); out.put(bytes);
    }
}
