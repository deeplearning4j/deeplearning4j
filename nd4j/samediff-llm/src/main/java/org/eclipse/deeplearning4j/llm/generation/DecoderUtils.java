/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.generation;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Delegation facade — all logic has moved to {@link ModelIOConfig},
 * {@link DecoderInputBuilder}, and {@link SameDiffMemoryUtils}.
 *
 * <p>Kept only so that existing test code compiles without mass-renaming.
 * Production code should reference the new classes directly.</p>
 *
 * @deprecated Use {@link ModelIOConfig}, {@link DecoderInputBuilder}, or
 *             {@link SameDiffMemoryUtils} instead.
 */
@Deprecated
public class DecoderUtils {

    /** @deprecated Use {@link ModelIOConfig#MASK_FILL} */
    @Deprecated
    public static final float MASK_FILL = ModelIOConfig.MASK_FILL;

    /** @deprecated Use {@link ModelIOConfig.KVCacheNames} */
    @Deprecated
    public static class KVCacheNames extends ModelIOConfig.KVCacheNames {
        public KVCacheNames(List<String> keyNames, List<String> valueNames) {
            super(keyNames, valueNames);
        }
    }

    // ========== Delegations to ModelIOConfig ==========

    /** @deprecated Use {@link ModelIOConfig#buildCausalMask(long, long)} */
    @Deprecated
    public static INDArray buildCausalMask(long currentSeqLen, long totalSeqLen) {
        return ModelIOConfig.buildCausalMask(currentSeqLen, totalSeqLen);
    }

    /** @deprecated Use {@link ModelIOConfig#buildCausalMask(long, long, long)} */
    @Deprecated
    public static INDArray buildCausalMask(long batchSize, long currentSeqLen, long totalSeqLen) {
        return ModelIOConfig.buildCausalMask(batchSize, currentSeqLen, totalSeqLen);
    }

    /** @deprecated Use {@link ModelIOConfig#createEmptyKvCache(SameDiff, String, long, long)} */
    @Deprecated
    public static INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize, long hiddenSize) {
        return ModelIOConfig.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize);
    }

    /** @deprecated Use {@link ModelIOConfig#createEmptyKvCache(SameDiff, String, long, long)} with seqLen=0 */
    @Deprecated
    public static INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize,
                                              long hiddenSize, long seqLen) {
        // The seqLen variant was only used for probe tests. For seqLen=0, delegate normally.
        // For seqLen>0, create a zeros buffer with the inferred shape.
        if (seqLen == 0) {
            return ModelIOConfig.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize);
        }
        // Probe path: create with the given seqLen. Infer shape via a zero-length buffer first.
        INDArray empty = ModelIOConfig.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize);
        long numHeads = empty.size(1);
        long headDim = empty.size(3);
        var kvType = empty.dataType();
        empty.close();
        return org.nd4j.linalg.factory.Nd4j.zeros(kvType, batchSize, numHeads, seqLen, headDim);
    }

    /** @deprecated Use {@link ModelIOConfig#findLogitsOutputName(SameDiff)} */
    @Deprecated
    public static String findLogitsOutputName(SameDiff decoder) {
        return ModelIOConfig.findLogitsOutputName(decoder);
    }

    /** @deprecated Use {@link ModelIOConfig#findKVCacheOutputNames(SameDiff)} */
    @Deprecated
    public static KVCacheNames findKVCacheOutputNames(SameDiff decoder) {
        ModelIOConfig.KVCacheNames parent = ModelIOConfig.findKVCacheOutputNames(decoder);
        return new KVCacheNames(parent.keyNames, parent.valueNames);
    }

    // ========== Delegations to DecoderInputBuilder ==========

    /** @deprecated Use {@link DecoderInputBuilder#buildDecoderInputMap} */
    @Deprecated
    public static Map<String, INDArray> buildDecoderInputMap(
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize) {
        return DecoderInputBuilder.buildDecoderInputMap(decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize);
    }

    /** @deprecated Use {@link DecoderInputBuilder#buildDecoderInputMap} */
    @Deprecated
    public static Map<String, INDArray> buildDecoderInputMap(
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs,
            boolean dspActive) {
        return DecoderInputBuilder.buildDecoderInputMap(decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, reusableInputs, dspActive);
    }

    /** @deprecated Use {@link DecoderInputBuilder#buildDecoderInputMap} */
    @Deprecated
    public static Map<String, INDArray> buildDecoderInputMap(
            ModelIOConfig ioConfig,
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs,
            boolean dspActive) {
        return DecoderInputBuilder.buildDecoderInputMap(ioConfig, decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, reusableInputs, dspActive);
    }

    /** @deprecated Use {@link DecoderInputBuilder#buildDecoderInputMap} */
    @Deprecated
    public static Map<String, INDArray> buildDecoderInputMap(
            ModelIOConfig ioConfig,
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs,
            boolean dspActive,
            INDArray encoderOutputs, INDArray encoderAttentionMask) {
        return DecoderInputBuilder.buildDecoderInputMap(ioConfig, decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, reusableInputs, dspActive,
                encoderOutputs, encoderAttentionMask);
    }

    /** @deprecated Use {@link DecoderInputBuilder#associateInternalModelInputs} */
    @Deprecated
    public static void associateInternalModelInputs(ModelIOConfig ioConfig,
                                                    List<String> materializedInputNames,
                                                    SameDiff decoder,
                                                    Map<String, INDArray> decoderInputMap) {
        DecoderInputBuilder.associateInternalModelInputs(ioConfig, materializedInputNames, decoder, decoderInputMap);
    }

    // ========== Scatter methods — kept inline since tests use them directly ==========

    /** Scatter new KV entries from decoder output into static KV buffers. */
    @Deprecated
    public static void scatterNewKvEntries(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long cachePos) {
        scatterNewKvEntries(staticKvBuffers, decoderOutputs, presentKeyNames, presentValueNames,
                maxKvLen, cachePos, ModelIOConfig.builder().build());
    }

    /** Scatter new KV entries with ModelIOConfig for name mapping. */
    @Deprecated
    public static void scatterNewKvEntries(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long cachePos,
            ModelIOConfig ioConfig) {

        java.util.List<INDArray> staticList = new java.util.ArrayList<>();
        java.util.List<INDArray> presentList = new java.util.ArrayList<>();

        java.util.List<String> allPresentNames = new java.util.ArrayList<>();
        allPresentNames.addAll(presentKeyNames);
        allPresentNames.addAll(presentValueNames);

        for (String presentName : allPresentNames) {
            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = ioConfig.presentToInputName(presentName);
            INDArray staticBuf = staticKvBuffers.get(pastInputName);
            if (staticBuf != null) {
                if (staticBuf.wasClosed() || staticBuf.data() == null) continue;
                if (presentKv.wasClosed() || presentKv.data() == null) continue;
                staticList.add(staticBuf);
                presentList.add(presentKv);
            }
        }

        if (staticList.isEmpty()) return;

        org.nd4j.linalg.factory.Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(
                staticList.toArray(new INDArray[0]),
                presentList.toArray(new INDArray[0]),
                cachePos));
    }

    /** Scatter a specific source position from present KV outputs into the static KV buffer. */
    @Deprecated
    public static void scatterNewKvEntryAtPosition(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long targetCachePos,
            int sourcePosition) {
        scatterNewKvEntryAtPosition(staticKvBuffers, decoderOutputs, presentKeyNames,
                presentValueNames, maxKvLen, targetCachePos, sourcePosition,
                ModelIOConfig.builder().build());
    }

    /** Scatter a specific source position with ModelIOConfig for name mapping. */
    @Deprecated
    public static void scatterNewKvEntryAtPosition(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long targetCachePos,
            int sourcePosition,
            ModelIOConfig ioConfig) {

        java.util.List<String> allPresentNames = new java.util.ArrayList<>();
        allPresentNames.addAll(presentKeyNames);
        allPresentNames.addAll(presentValueNames);

        for (String presentName : allPresentNames) {
            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = ioConfig.presentToInputName(presentName);
            INDArray staticBuf = staticKvBuffers.get(pastInputName);
            if (staticBuf == null) continue;

            INDArray sourceSlice = presentKv.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(sourcePosition), org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray destSlice = staticBuf.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(targetCachePos), org.nd4j.linalg.indexing.NDArrayIndex.all());
            destSlice.assign(sourceSlice);
        }
    }

    /** Scatter multiple accepted positions from a speculative decode step. */
    @Deprecated
    public static void scatterMultipleKvEntries(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long baseCachePos,
            int numPositions) {
        scatterMultipleKvEntries(staticKvBuffers, decoderOutputs, presentKeyNames,
                presentValueNames, maxKvLen, baseCachePos, numPositions,
                ModelIOConfig.builder().build());
    }

    /** Scatter multiple positions with ModelIOConfig for name mapping. */
    @Deprecated
    public static void scatterMultipleKvEntries(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long baseCachePos,
            int numPositions,
            ModelIOConfig ioConfig) {

        if (numPositions <= 0) return;

        java.util.List<String> allPresentNames = new java.util.ArrayList<>();
        allPresentNames.addAll(presentKeyNames);
        allPresentNames.addAll(presentValueNames);

        for (String presentName : allPresentNames) {
            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = ioConfig.presentToInputName(presentName);
            INDArray staticBuf = staticKvBuffers.get(pastInputName);
            if (staticBuf == null) continue;

            INDArray sourceSlice = presentKv.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(maxKvLen, maxKvLen + numPositions),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray destSlice = staticBuf.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(baseCachePos, baseCachePos + numPositions),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            destSlice.assign(sourceSlice);
        }
    }

    /** Pad KV cache to static size (delegation — kept for test compatibility). */
    @Deprecated
    public static Map<String, INDArray> padKvCacheToStaticSize(
            Map<String, INDArray> prefillOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen) {
        return padKvCacheToStaticSize(prefillOutputs, presentKeyNames, presentValueNames,
                maxKvLen, ModelIOConfig.builder().build());
    }

    /** Pad KV cache to static size with ModelIOConfig. */
    @Deprecated
    public static Map<String, INDArray> padKvCacheToStaticSize(
            Map<String, INDArray> prefillOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            ModelIOConfig ioConfig) {

        java.util.Map<String, INDArray> staticKvBuffers = new java.util.HashMap<>();
        java.util.List<String> allPresentNames = new java.util.ArrayList<>();
        allPresentNames.addAll(presentKeyNames);
        allPresentNames.addAll(presentValueNames);

        org.nd4j.linalg.factory.Nd4j.getExecutioner().commit();

        for (String presentName : allPresentNames) {
            INDArray prefillKv = prefillOutputs.get(presentName);
            if (prefillKv == null) continue;

            long[] shape = prefillKv.shape();
            long batch = shape[0], heads = shape[1], prefillLen = shape[2], dim = shape[3];
            long padLen = maxKvLen - prefillLen;

            INDArray staticBuf;
            if (padLen > 0) {
                INDArray padding = org.nd4j.linalg.factory.Nd4j.zeros(prefillKv.dataType(), batch, heads, padLen, dim);
                staticBuf = org.nd4j.linalg.factory.Nd4j.concat(2, prefillKv, padding);
                padding.close();
            } else {
                staticBuf = prefillKv.dup();
            }

            String pastInputName = ioConfig.presentToInputName(presentName);
            staticBuf.setCloseable(false);
            staticKvBuffers.put(pastInputName, staticBuf);
        }

        return staticKvBuffers;
    }

    /** @deprecated Use {@link SameDiffMemoryUtils#trimAllDevicePools()} */
    @Deprecated
    public static void trimAllDevicePools() {
        SameDiffMemoryUtils.trimAllDevicePools();
    }
}
