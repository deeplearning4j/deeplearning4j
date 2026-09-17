/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm;

import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Collection;
import java.util.Map;

/** Reader for DspDiagnostics' DSPTS001 artifacts. Integers are little endian;
 * payload endianness is explicit. Each tensor payload is dense C-logical storage,
 * while shape/strides/order/base offset describe the source. Checksums are FNV-1a
 * over the complete raw payload, not samples. Reconstruction owns independent
 * storage with original strides/order and a normalized base offset (no unrelated
 * parent bytes). No downloads, graph execution, or weights are involved here.
 */
final class DspTensorSnapshot {
    static final List<String> MTP_INPUTS = List.of("mtp_input_ids", "mtp_target_hidden_states",
            "mtp_position_offset", "mtp_cache_position", "mtp_causal_mask",
            "mtp_past_key_values.0.key", "mtp_past_key_values.0.value");
    static final String LOGITS = "output/pre-argmax/mtp_logits";
    static final String HIDDEN = "output/pre-argmax/mtp_hidden";
    static final String DRAFT = "output/post-argmax/draft_id";
    static final int BUDGET = 32 * 1024 * 1024;
    static final int MAX_METADATA = 128 * 1024;
    final int callIndex;
    final long sourcePosition;
    final String source;
    final ByteOrder payloadOrder;
    final Map<String, Tensor> tensors = new LinkedHashMap<>();

    static final class Tensor {
        String name;
        DataType dtype;
        int width;
        char order;
        long originalOffset;
        long[] shape, strides;
        long checksum;
        byte[] raw;

        long logicalOffset(long index) {
            long offset = 0;
            for (int d = shape.length - 1; d >= 0; d--) {
                offset = Math.addExact(offset, Math.multiplyExact(index % shape[d], strides[d]));
                index /= shape[d];
            }
            return offset;
        }

        void restore(INDArray result, ByteOrder payloadOrder) {
            if (result.dataType() != dtype || !Arrays.equals(result.shape(), shape)
                    || !Arrays.equals(result.stride(), strides))
                throw new IllegalArgumentException("Snapshot destination metadata mismatch: " + name);
            if (raw.length == 0) return;
            // Borrow host pointer; never close it. Preserve all storage bits without casts.
            BytePointer ptr = new BytePointer(result.data().pointer()).capacity(result.data().length() * width);
            byte[] element = new byte[width];
            for (int i = 0; i < raw.length / width; i++) {
                for (int b = 0; b < width; b++) {
                    int sourceByte = payloadOrder == ByteOrder.nativeOrder() ? b : width - 1 - b;
                    element[b] = raw[i * width + sourceByte];
                }
                ptr.position((logicalOffset(i) + result.offset()) * width).put(element);
            }
            Nd4j.getAffinityManager().tagLocation(result, AffinityManager.Location.HOST);
        }

        INDArray reconstruct(ByteOrder payloadOrder) {
            try (MemoryWorkspace ignored = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                if (raw.length == 0) return Nd4j.createUninitialized(dtype, shape, order);
                long min = 0, max = 0;
                for (int d = 0; d < shape.length; d++) {
                    long delta = Math.multiplyExact(shape[d] - 1, strides[d]);
                    min = Math.addExact(min, Math.min(delta, 0));
                    max = Math.addExact(max, Math.max(delta, 0));
                }
                long length = Math.addExact(Math.subtractExact(max, min), 1);
                if (Math.multiplyExact(length, width) > BUDGET)
                    throw new IllegalArgumentException("Snapshot layout span exceeds reconstruction budget: " + name);
                DataBuffer buffer = Nd4j.createBuffer(dtype, length, true);
                INDArray result;
                try {
                    result = Nd4j.create(buffer, shape, strides, -min, order, dtype);
                } catch (RuntimeException | Error failure) { buffer.close(); throw failure; }
                try {
                    restore(result, payloadOrder);
                    return result;
                } catch (RuntimeException | Error failure) { result.close(); throw failure; }
            }
        }
    }

    private DspTensorSnapshot(int callIndex, long sourcePosition, String source, ByteOrder payloadOrder) {
        this.callIndex = callIndex; this.sourcePosition = sourcePosition;
        this.source = source; this.payloadOrder = payloadOrder;
    }

    static DspTensorSnapshot read(Path path) throws IOException {
        long size = Files.size(path);
        if (size > BUDGET + MAX_METADATA || size < 8) throw new IOException("Invalid snapshot size: " + size);
        // Bounded even if a producer is still writing; missing footer/length is rejected.
        try (var input = Files.newInputStream(path)) {
            byte[] bytes = input.readNBytes(BUDGET + MAX_METADATA + 1);
            if (bytes.length > BUDGET + MAX_METADATA) throw new IOException("Snapshot exceeds budget");
            return decode(bytes);
        }
    }

    static DspTensorSnapshot decode(byte[] bytes) throws IOException {
        if (bytes.length > BUDGET + MAX_METADATA) throw new IOException("Snapshot exceeds budget");
        try {
            ByteBuffer in = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            require(magic(in, "DSPTS001"), "Snapshot magic/version");
            int call = in.getInt(); long position = in.getLong(); String source = string(in);
            require(call >= 1 && call <= 3 && !source.isEmpty(), "Snapshot call/source");
            int count = in.getInt(); require(count > 0 && count <= 64, "Snapshot tensor count");
            int endian = in.get() & 255; require(endian <= 1, "Snapshot byte order");
            DspTensorSnapshot snapshot = new DspTensorSnapshot(call, position, source,
                    endian == 1 ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
            Map<Tensor, Integer> sizes = new LinkedHashMap<>();
            long total = 0;
            for (int n = 0; n < count; n++) {
                Tensor t = new Tensor(); t.name = string(in); t.dtype = DataType.fromInt(in.getInt());
                t.width = in.getInt(); int rank = in.getInt(); t.order = (char) in.get();
                t.originalOffset = in.getLong();
                require(!t.name.isEmpty() && !snapshot.tensors.containsKey(t.name), "Snapshot duplicate/empty name");
                require(rank >= 0 && rank <= 32 && (t.order == 'c' || t.order == 'f'), "Snapshot rank/order");
                require(t.originalOffset >= 0 && t.width >= 1 && t.width <= 8 && t.width == t.dtype.width()
                        && (t.dtype.isNumerical() || t.dtype == DataType.BOOL), "Snapshot dtype/offset");
                t.shape = new long[rank]; t.strides = new long[rank];
                long elements = 1;
                for (int d = 0; d < rank; d++) {
                    t.shape[d] = in.getLong(); t.strides[d] = in.getLong();
                    require(t.shape[d] >= 0, "Snapshot negative shape");
                    elements = Math.multiplyExact(elements, t.shape[d]);
                }
                long length = in.getLong(); t.checksum = in.getLong();
                require(length >= 0 && length == Math.multiplyExact(elements, t.width), "Snapshot payload shape/size");
                total = Math.addExact(total, length); require(total <= BUDGET, "Snapshot payload budget");
                snapshot.tensors.put(t.name, t); sizes.put(t, Math.toIntExact(length));
            }
            require(in.position() <= MAX_METADATA && in.remaining() == total + 8, "Incomplete snapshot or trailing bytes");
            for (Map.Entry<Tensor, Integer> entry : sizes.entrySet()) {
                Tensor t = entry.getKey(); t.raw = new byte[entry.getValue()]; in.get(t.raw);
                require(checksum(t.raw) == t.checksum, "Snapshot checksum mismatch: " + t.name);
            }
            require(magic(in, "DSPTDONE") && !in.hasRemaining(), "Incomplete snapshot footer");
            return snapshot;
        } catch (RuntimeException failure) { throw new IOException("Malformed DSP tensor snapshot", failure); }
    }

    Map<String, INDArray> reconstruct() {
        return reconstruct(tensors.keySet());
    }

    Map<String, INDArray> reconstructInputs() {
        if (!tensors.keySet().containsAll(MTP_INPUTS))
            throw new IllegalArgumentException("Snapshot missing dynamic MTP inputs");
        return reconstruct(MTP_INPUTS);
    }

    private Map<String, INDArray> reconstruct(Collection<String> names) {
        Map<String, INDArray> result = new LinkedHashMap<>();
        try {
            long storageBytes = 0;
            for (String name : names) {
                Tensor t = tensors.get(name);
                long low = 0, high = 0;
                if (t.raw.length != 0) {
                    for (int d = 0; d < t.shape.length; d++) {
                        long delta = Math.multiplyExact(t.shape[d] - 1, t.strides[d]);
                        low = Math.addExact(low, Math.min(0, delta));
                        high = Math.addExact(high, Math.max(0, delta));
                    }
                    storageBytes = Math.addExact(storageBytes,
                            Math.multiplyExact(Math.addExact(Math.subtractExact(high, low), 1), t.width));
                }
                if (storageBytes > BUDGET) throw new IllegalArgumentException("Snapshot reconstructed storage exceeds 32 MiB");
            }
            for (String name : names) result.put(name, tensors.get(name).reconstruct(payloadOrder));
            return result;
        } catch (RuntimeException | Error failure) {
            for (INDArray a : result.values()) a.close();
            throw failure;
        }
    }

    static long checksum(byte[] bytes) {
        long hash = 0xcbf29ce484222325L;
        for (byte b : bytes) { hash ^= b & 255; hash *= 0x100000001b3L; }
        return hash;
    }
    private static String string(ByteBuffer in) throws IOException {
        int length = in.getInt(); require(length >= 0 && length <= 1024 && length <= in.remaining(), "Snapshot string size");
        byte[] bytes = new byte[length]; in.get(bytes); return new String(bytes, StandardCharsets.UTF_8);
    }
    private static boolean magic(ByteBuffer in, String value) {
        byte[] bytes = new byte[8]; in.get(bytes); return Arrays.equals(bytes, value.getBytes(StandardCharsets.US_ASCII));
    }
    private static void require(boolean condition, String message) throws IOException {
        if (!condition) throw new IOException(message);
    }
}
