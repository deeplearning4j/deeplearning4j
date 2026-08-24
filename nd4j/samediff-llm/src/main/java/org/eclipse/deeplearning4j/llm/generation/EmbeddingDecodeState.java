/*
 *  ******************************************************************************
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/** Caller-owned fixed-buffer inputs retained by the VLM/prebuilt-embedding decode path. */
@Slf4j
final class EmbeddingDecodeState implements AutoCloseable {
    private final Map<String, INDArray> staticKvBuffers;
    private final INDArray[] decodeInputs;
    private boolean closed;

    EmbeddingDecodeState(Map<String, INDArray> staticKvBuffers, INDArray... decodeInputs) {
        this.staticKvBuffers = staticKvBuffers == null
                ? Map.of() : new LinkedHashMap<>(staticKvBuffers);
        this.decodeInputs = decodeInputs == null ? new INDArray[0] : decodeInputs.clone();
    }

    boolean matches(Map<String, INDArray> candidateKvBuffers, INDArray... candidateInputs) {
        return mismatch(candidateKvBuffers, candidateInputs) == null;
    }

    String mismatch(Map<String, INDArray> candidateKvBuffers, INDArray... candidateInputs) {
        Map<String, INDArray> candidates = candidateKvBuffers == null ? Map.of() : candidateKvBuffers;
        if (!staticKvBuffers.keySet().equals(candidates.keySet())) {
            return "KV keys retained=" + staticKvBuffers.keySet() + ", candidate=" + candidates.keySet();
        }
        for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
            INDArray candidate = candidates.get(entry.getKey());
            if (entry.getValue() != candidate) {
                return "KV '" + entry.getKey() + "' " + identity(entry.getValue(), candidate);
            }
        }
        INDArray[] inputs = candidateInputs == null ? new INDArray[0] : candidateInputs;
        if (decodeInputs.length != inputs.length) {
            return "decode input count retained=" + decodeInputs.length + ", candidate=" + inputs.length;
        }
        for (int i = 0; i < decodeInputs.length; i++) {
            if (decodeInputs[i] != inputs[i]) {
                return "decode input[" + i + "] " + identity(decodeInputs[i], inputs[i]);
            }
        }
        return null;
    }

    private static String identity(INDArray retained, INDArray candidate) {
        return "retained=" + describe(retained) + ", candidate=" + describe(candidate);
    }

    private static String describe(INDArray array) {
        if (array == null) return "null";
        Object data = array.data();
        return "array@" + Integer.toHexString(System.identityHashCode(array))
                + "/data@" + Integer.toHexString(System.identityHashCode(data))
                + "/shape=" + Arrays.toString(array.shape())
                + "/closed=" + array.wasClosed();
    }

    INDArray kvBuffer(String name) {
        return staticKvBuffers.get(name);
    }

    INDArray decodeInput(int index) {
        return index >= 0 && index < decodeInputs.length ? decodeInputs[index] : null;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        Set<INDArray> owned = Collections.newSetFromMap(new IdentityHashMap<>());
        owned.addAll(staticKvBuffers.values());
        owned.addAll(Arrays.asList(decodeInputs));
        for (INDArray array : owned) closeOwned(array);
    }

    private static void closeOwned(INDArray array) {
        if (array == null || array.wasClosed()) return;
        try {
            // Static KV buffers are pinned while replay is live. The decoder session must be
            // retired before this owner is closed. SameDiff may mark placeholder buffers
            // constant and attach opaque handles, so plain INDArray.close() can be a no-op;
            // use the shared authoritative close path that clears both protections.
            SameDiffMemoryUtils.safeClose(array);
        } catch (Exception failure) {
            log.warn("Error closing retained embedding decode input: {}", failure.getMessage());
        }
    }
}
