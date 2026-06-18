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

package org.eclipse.deeplearning4j.llm.tokenizer;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Represents the result of tokenizing text.
 *
 * An Encoding contains:
 * - Token IDs: integer indices into the vocabulary
 * - Tokens: the actual token strings
 * - Attention mask: 1 for real tokens, 0 for padding
 * - Type IDs: segment IDs for multi-sequence inputs
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * Encoding encoding = tokenizer.encode("Hello, world!", true);
 *
 * // Get token IDs as array
 * int[] ids = encoding.getIds();
 *
 * // Get as INDArray for model input
 * INDArray inputIds = encoding.getIdsAsArray();
 * INDArray attentionMask = encoding.getAttentionMaskAsArray();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
public class Encoding {

    /**
     * The token IDs (indices into vocabulary).
     */
    private final int[] ids;

    /**
     * The token strings.
     */
    private final String[] tokens;

    /**
     * Attention mask: 1 for real tokens, 0 for padding.
     */
    private final int[] attentionMask;

    /**
     * Type IDs for multi-sequence inputs (e.g., sentence A vs B).
     */
    private final int[] typeIds;

    /**
     * Get the number of tokens in this encoding.
     *
     * @return the token count
     */
    public int getLength() {
        return ids != null ? ids.length : 0;
    }

    /**
     * Get token IDs as an INDArray.
     *
     * @return 1D INDArray of token IDs with shape [length]
     */
    public INDArray getIdsAsArray() {
        return Nd4j.createFromArray(ids);
    }

    /**
     * Get token IDs as a batched INDArray.
     *
     * @return 2D INDArray of token IDs with shape [1, length]
     */
    public INDArray getIdsAsBatchedArray() {
        return Nd4j.createFromArray(ids).reshape(1, ids.length);
    }

    /**
     * Get attention mask as an INDArray.
     *
     * @return 1D INDArray of attention mask with shape [length]
     */
    public INDArray getAttentionMaskAsArray() {
        return Nd4j.createFromArray(attentionMask);
    }

    /**
     * Get attention mask as a batched INDArray.
     *
     * @return 2D INDArray of attention mask with shape [1, length]
     */
    public INDArray getAttentionMaskAsBatchedArray() {
        return Nd4j.createFromArray(attentionMask).reshape(1, attentionMask.length);
    }

    /**
     * Get type IDs as an INDArray.
     *
     * @return 1D INDArray of type IDs with shape [length]
     */
    public INDArray getTypeIdsAsArray() {
        return Nd4j.createFromArray(typeIds);
    }

    /**
     * Get type IDs as a batched INDArray.
     *
     * @return 2D INDArray of type IDs with shape [1, length]
     */
    public INDArray getTypeIdsAsBatchedArray() {
        return Nd4j.createFromArray(typeIds).reshape(1, typeIds.length);
    }

    /**
     * Pad this encoding to the specified length.
     *
     * @param maxLength the target length
     * @param padTokenId the token ID to use for padding
     * @param padOnRight whether to pad on the right (true) or left (false)
     * @return a new Encoding with padding applied
     */
    public Encoding pad(int maxLength, int padTokenId, boolean padOnRight) {
        if (ids.length >= maxLength) {
            return this;
        }

        int padLength = maxLength - ids.length;

        int[] paddedIds = new int[maxLength];
        int[] paddedMask = new int[maxLength];
        int[] paddedTypeIds = new int[maxLength];
        String[] paddedTokens = new String[maxLength];

        if (padOnRight) {
            System.arraycopy(ids, 0, paddedIds, 0, ids.length);
            System.arraycopy(attentionMask, 0, paddedMask, 0, attentionMask.length);
            System.arraycopy(typeIds, 0, paddedTypeIds, 0, typeIds.length);
            System.arraycopy(tokens, 0, paddedTokens, 0, tokens.length);

            for (int i = ids.length; i < maxLength; i++) {
                paddedIds[i] = padTokenId;
                paddedMask[i] = 0;
                paddedTypeIds[i] = 0;
                paddedTokens[i] = "[PAD]";
            }
        } else {
            for (int i = 0; i < padLength; i++) {
                paddedIds[i] = padTokenId;
                paddedMask[i] = 0;
                paddedTypeIds[i] = 0;
                paddedTokens[i] = "[PAD]";
            }

            System.arraycopy(ids, 0, paddedIds, padLength, ids.length);
            System.arraycopy(attentionMask, 0, paddedMask, padLength, attentionMask.length);
            System.arraycopy(typeIds, 0, paddedTypeIds, padLength, typeIds.length);
            System.arraycopy(tokens, 0, paddedTokens, padLength, tokens.length);
        }

        return Encoding.builder()
                .ids(paddedIds)
                .tokens(paddedTokens)
                .attentionMask(paddedMask)
                .typeIds(paddedTypeIds)
                .build();
    }

    /**
     * Truncate this encoding to the specified length.
     *
     * @param maxLength the maximum length
     * @param truncateFromRight whether to truncate from the right (true) or left (false)
     * @return a new Encoding with truncation applied
     */
    public Encoding truncate(int maxLength, boolean truncateFromRight) {
        if (ids.length <= maxLength) {
            return this;
        }

        int[] truncatedIds = new int[maxLength];
        int[] truncatedMask = new int[maxLength];
        int[] truncatedTypeIds = new int[maxLength];
        String[] truncatedTokens = new String[maxLength];

        if (truncateFromRight) {
            System.arraycopy(ids, 0, truncatedIds, 0, maxLength);
            System.arraycopy(attentionMask, 0, truncatedMask, 0, maxLength);
            System.arraycopy(typeIds, 0, truncatedTypeIds, 0, maxLength);
            System.arraycopy(tokens, 0, truncatedTokens, 0, maxLength);
        } else {
            int offset = ids.length - maxLength;
            System.arraycopy(ids, offset, truncatedIds, 0, maxLength);
            System.arraycopy(attentionMask, offset, truncatedMask, 0, maxLength);
            System.arraycopy(typeIds, offset, truncatedTypeIds, 0, maxLength);
            System.arraycopy(tokens, offset, truncatedTokens, 0, maxLength);
        }

        return Encoding.builder()
                .ids(truncatedIds)
                .tokens(truncatedTokens)
                .attentionMask(truncatedMask)
                .typeIds(truncatedTypeIds)
                .build();
    }

    // =========================================================================
    // Device-Aware Methods for Multi-Chip Backend Support
    // =========================================================================

    /**
     * Get token IDs as an INDArray on a specific device.
     * Ensures the data is transferred to the target device.
     *
     * @param device the target device descriptor
     * @return 1D INDArray of token IDs on the specified device
     */
    public INDArray getIdsAsArrayOnDevice(DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(ids);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Get token IDs as a batched INDArray on a specific device.
     *
     * @param device the target device descriptor
     * @return 2D INDArray of token IDs with shape [1, length] on the specified device
     */
    public INDArray getIdsAsBatchedArrayOnDevice(DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(ids).reshape(1, ids.length);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Get attention mask as an INDArray on a specific device.
     *
     * @param device the target device descriptor
     * @return 1D INDArray of attention mask on the specified device
     */
    public INDArray getAttentionMaskAsArrayOnDevice(DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(attentionMask);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Get attention mask as a batched INDArray on a specific device.
     *
     * @param device the target device descriptor
     * @return 2D INDArray of attention mask with shape [1, length] on the specified device
     */
    public INDArray getAttentionMaskAsBatchedArrayOnDevice(DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(attentionMask).reshape(1, attentionMask.length);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Get type IDs as an INDArray on a specific device.
     *
     * @param device the target device descriptor
     * @return 1D INDArray of type IDs on the specified device
     */
    public INDArray getTypeIdsAsArrayOnDevice(DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(typeIds);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Get all encoding arrays (ids, attention mask, type ids) as a map on a specific device.
     * This is useful for passing directly to model inputs.
     *
     * @param device the target device descriptor
     * @return map with "input_ids", "attention_mask", and "token_type_ids" keys
     */
    public java.util.Map<String, INDArray> getAsModelInputsOnDevice(DeviceDescriptor device) {
        java.util.Map<String, INDArray> inputs = new java.util.HashMap<>();
        inputs.put("input_ids", getIdsAsBatchedArrayOnDevice(device));
        inputs.put("attention_mask", getAttentionMaskAsBatchedArrayOnDevice(device));
        if (typeIds != null) {
            inputs.put("token_type_ids", getTypeIdsAsArrayOnDevice(device).reshape(1, typeIds.length));
        }
        return inputs;
    }

    /**
     * Get token IDs within a workspace on a specific device.
     * Uses workspace-based allocation for efficient memory reuse.
     *
     * @param workspace the multi-backend workspace
     * @param device the target device
     * @return INDArray allocated within the workspace on the target device
     */
    public INDArray getIdsInWorkspace(MultiBackendWorkspace workspace, DeviceDescriptor device) {
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            INDArray array = Nd4j.createFromArray(ids).reshape(1, ids.length);
            ensureOnDevice(array, device);
            return array;
        }
    }

    /**
     * Prefetch encoding data to a device asynchronously.
     * This hints that the data will be needed on the device soon.
     *
     * @param device the target device
     */
    public void prefetchToDevice(DeviceDescriptor device) {
        // Create arrays and trigger async prefetch
        INDArray idsArray = Nd4j.createFromArray(ids);
        INDArray maskArray = Nd4j.createFromArray(attentionMask);

        DataBuffer idBuffer = idsArray.data();
        DataBuffer maskBuffer = maskArray.data();

        if (idBuffer.isHybrid()) {
            idBuffer.asHybrid().prefetch(device);
        }
        if (maskBuffer.isHybrid()) {
            maskBuffer.asHybrid().prefetch(device);
        }
    }

    /**
     * Ensure an INDArray is available on the specified device.
     * Triggers data transfer if necessary.
     *
     * @param array the array to transfer
     * @param device the target device
     */
    private void ensureOnDevice(INDArray array, DeviceDescriptor device) {
        if (device == null) {
            return;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().ensureAvailableOn(device);
        }
    }
}
