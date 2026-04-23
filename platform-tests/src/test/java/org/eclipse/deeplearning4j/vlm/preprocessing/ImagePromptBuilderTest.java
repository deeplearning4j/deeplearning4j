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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class ImagePromptBuilderTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildNoTiling(Nd4jBackend backend) {
        // rows=0, cols=0 → global-only format
        String prompt = ImagePromptBuilder.buildImagePromptString(0, 0, 3);

        assertTrue(prompt.contains("<global-img>"));
        assertTrue(prompt.contains("<fake_token_around_image>"));

        // Should have exactly 3 <image> tokens
        int imageCount = countSubstring(prompt, "<image>");
        assertEquals(3, imageCount);

        // Should NOT have row/col tokens
        assertFalse(prompt.contains("<row_"));
        assertFalse(prompt.contains("_col_"));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildWithTiling(Nd4jBackend backend) {
        // 2 rows, 3 cols, 2 image tokens per tile
        String prompt = ImagePromptBuilder.buildImagePromptString(2, 3, 2);

        // Global frame should be emitted before tile descriptors in packed layout
        assertTrue(prompt.indexOf("<global-img>") < prompt.indexOf("<row_1_col_1>"));

        // Should have row/col tokens for each tile
        assertTrue(prompt.contains("<row_1_col_1>"));
        assertTrue(prompt.contains("<row_1_col_2>"));
        assertTrue(prompt.contains("<row_1_col_3>"));
        assertTrue(prompt.contains("<row_2_col_1>"));
        assertTrue(prompt.contains("<row_2_col_2>"));
        assertTrue(prompt.contains("<row_2_col_3>"));

        // Should have global section
        assertTrue(prompt.contains("<global-img>"));

        // Total <image> tokens: (2*3 tiles + 1 global) * 2 tokens = 14
        int imageCount = countSubstring(prompt, "<image>");
        assertEquals(14, imageCount);

        // Packed global-first layout should still preserve descriptor order
        assertTrue(prompt.indexOf("<row_1_col_1>") < prompt.indexOf("<row_1_col_2>"));
        assertTrue(prompt.indexOf("<row_1_col_2>") < prompt.indexOf("<row_1_col_3>"));
        assertTrue(prompt.indexOf("<row_1_col_3>") < prompt.indexOf("<row_2_col_1>"));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testImageTokenCount(Nd4jBackend backend) {
        int seqLen = 169; // typical for 512x512 vision encoder with patch size 14
        String prompt = ImagePromptBuilder.buildImagePromptString(2, 2, seqLen);

        // (2*2 tiles + 1 global) * 169 = 845 <image> tokens
        int imageCount = countSubstring(prompt, "<image>");
        assertEquals(5 * seqLen, imageCount);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCountOccurrences(Nd4jBackend backend) {
        int[] ids = {1, 2, 3, 2, 4, 2, 5};
        assertEquals(3, ImagePromptBuilder.countOccurrences(ids, 2));
        assertEquals(1, ImagePromptBuilder.countOccurrences(ids, 1));
        assertEquals(0, ImagePromptBuilder.countOccurrences(ids, 99));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExpandImageTokens(Nd4jBackend backend) {
        int imageTokenId = 100;
        int[] promptIds = {1, 2, imageTokenId, 3, 4};

        int[] expanded = ImagePromptBuilder.expandImageTokens(promptIds, imageTokenId, 5);

        // Original: [1, 2, 100, 3, 4] (len=5)
        // Expanded: [1, 2, 100, 100, 100, 100, 100, 3, 4] (len=9)
        assertEquals(9, expanded.length);
        assertEquals(1, expanded[0]);
        assertEquals(2, expanded[1]);
        for (int i = 2; i < 7; i++) {
            assertEquals(imageTokenId, expanded[i]);
        }
        assertEquals(3, expanded[7]);
        assertEquals(4, expanded[8]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExpandImageTokensNoImageToken(Nd4jBackend backend) {
        int[] promptIds = {1, 2, 3, 4};
        int[] result = ImagePromptBuilder.expandImageTokens(promptIds, 100, 5);

        // No image token found → return original
        assertSame(promptIds, result);
    }

    private static int countSubstring(String str, String sub) {
        int count = 0;
        int idx = 0;
        while ((idx = str.indexOf(sub, idx)) != -1) {
            count++;
            idx += sub.length();
        }
        return count;
    }
}
