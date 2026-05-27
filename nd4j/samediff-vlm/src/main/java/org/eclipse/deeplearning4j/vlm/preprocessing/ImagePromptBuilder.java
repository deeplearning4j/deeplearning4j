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

package org.eclipse.deeplearning4j.vlm.preprocessing;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

/**
 * Builds image prompt strings for SmolDocling-style tiled vision inputs.
 * Uses row/col tokens and a global image token when split into tiles.
 */
@Slf4j
public class ImagePromptBuilder {

    /**
     * Build the expanded image prompt string for tiled document pages.
     *
     * The prompt uses tiles-first, global-last ordering to match the HuggingFace
     * Idefics3 preprocessing that the model was trained with. Tile descriptors are
     * emitted in row-major order, with newlines between rows.
     *
     * @param imageRows number of tile rows (0 if no tiling)
     * @param imageCols number of tile columns (0 if no tiling)
     * @param imageSeqLen number of image tokens per tile
     * @return the expanded prompt string with image tokens
     */
    public static String buildImagePromptString(int imageRows, int imageCols, int imageSeqLen) {
        String fake = "<fake_token_around_image>";
        String image = "<image>";
        String global = "<global-img>";

        if (imageRows <= 0 || imageCols <= 0) {
            StringBuilder sb = new StringBuilder();
            sb.append(fake).append(global);
            for (int i = 0; i < imageSeqLen; i++) {
                sb.append(image);
            }
            sb.append(fake);
            return sb.toString();
        }

        // Tiles first in row-major order, then global last (matching HF Idefics3)
        StringBuilder sb = new StringBuilder();
        for (int r = 1; r <= imageRows; r++) {
            for (int c = 1; c <= imageCols; c++) {
                sb.append(fake).append("<row_" + r + "_col_" + c + ">");
                for (int j = 0; j < imageSeqLen; j++) {
                    sb.append(image);
                }
            }
            sb.append("\n");
        }
        // Global image last
        sb.append(fake).append(global);
        for (int j = 0; j < imageSeqLen; j++) {
            sb.append(image);
        }
        sb.append(fake);

        return sb.toString();
    }

    /**
     * Expand a single &lt;image&gt; token into a sequence that matches the vision token count.
     *
     * @param promptTokenIds the original token IDs
     * @param imageTokenId the token ID for &lt;image&gt;
     * @param imageTokenCount the number of image tokens to expand to
     * @return expanded token ID array
     */
    public static int[] expandImageTokens(int[] promptTokenIds, int imageTokenId, int imageTokenCount) {
        if (imageTokenCount <= 0) {
            return promptTokenIds;
        }
        int firstImagePos = -1;
        int occurrences = 0;
        for (int i = 0; i < promptTokenIds.length; i++) {
            if (promptTokenIds[i] == imageTokenId) {
                occurrences++;
                if (firstImagePos < 0) {
                    firstImagePos = i;
                }
            }
        }
        if (firstImagePos < 0) {
            log.warn("No <image> token found for expansion");
            return promptTokenIds;
        }

        int newLen = promptTokenIds.length - 1 + imageTokenCount;
        int[] expanded = new int[newLen];
        int outIdx = 0;
        for (int i = 0; i < promptTokenIds.length; i++) {
            int id = promptTokenIds[i];
            if (id == imageTokenId && i == firstImagePos) {
                for (int k = 0; k < imageTokenCount; k++) {
                    expanded[outIdx++] = imageTokenId;
                }
            } else {
                expanded[outIdx++] = id;
            }
        }

        if (occurrences > 1) {
            log.warn("Found {} <image> tokens; only expanded the first one", occurrences);
        }
        return expanded;
    }

    /**
     * Resolve the &lt;image&gt; token ID from the tokenizer.
     *
     * @param tokenizer the tokenizer to query
     * @return the token ID for &lt;image&gt;
     */
    public static int resolveImageTokenId(Tokenizer tokenizer) {
        Integer id = tokenizer.getTokenId("<image>");
        if (id != null && id >= 0) {
            return id;
        }
        int[] encoded = tokenizer.encode("<image>", false).getIds();
        if (encoded.length == 1) {
            return encoded[0];
        }
        log.warn("Could not resolve <image> token id from tokenizer; using fallback 49190");
        return 49190;
    }

    /**
     * Resolve the &lt;video&gt; token ID from the tokenizer.
     *
     * <p>Tries &lt;video&gt; first, then falls back to &lt;image&gt; since some
     * video VLMs (e.g., SmolVLM2) reuse the image token for video frames.</p>
     *
     * @param tokenizer the tokenizer to query
     * @return the token ID for &lt;video&gt; or &lt;image&gt; as fallback
     */
    public static int resolveVideoTokenId(Tokenizer tokenizer) {
        Integer id = tokenizer.getTokenId("<video>");
        if (id != null && id >= 0) {
            return id;
        }
        int[] encoded = tokenizer.encode("<video>", false).getIds();
        if (encoded.length == 1) {
            return encoded[0];
        }
        // Fall back to <image> token since many video VLMs reuse it
        log.info("No <video> token found, falling back to <image> token for video frames");
        return resolveImageTokenId(tokenizer);
    }

    /**
     * Count occurrences of a target ID in an array.
     *
     * @param ids the array to search
     * @param targetId the target value
     * @return number of occurrences
     */
    public static int countOccurrences(int[] ids, int targetId) {
        int count = 0;
        for (int id : ids) {
            if (id == targetId) {
                count++;
            }
        }
        return count;
    }
}
