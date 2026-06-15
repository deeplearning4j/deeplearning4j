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

package org.eclipse.deeplearning4j.vlm.model;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class EmbeddingMergerTest extends BaseNd4jTestWithBackends {

    private static final int IMAGE_TOKEN_ID = 100;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMergeReplacesImagePositions(Nd4jBackend backend) {
        int seqLen = 5;
        int hiddenDim = 4;
        int visionSeqLen = 2;

        // Text embeddings: [1, 5, 4] filled with 1.0
        INDArray textEmbeddings = Nd4j.ones(DataType.FLOAT, 1, seqLen, hiddenDim);

        // Vision embeddings: [1, 2, 4] filled with 9.0
        INDArray visionEmbeddings = Nd4j.valueArrayOf(new long[]{1, visionSeqLen, hiddenDim}, 9.0f);

        // Token IDs: positions 1 and 3 are <image> tokens
        int[] tokenIds = {1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3};

        INDArray merged = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, tokenIds, IMAGE_TOKEN_ID);

        assertArrayEquals(new long[]{1, seqLen, hiddenDim}, merged.shape());

        // Position 0: text (1.0)
        for (int d = 0; d < hiddenDim; d++) {
            assertEquals(1.0f, merged.getFloat(0, 0, d), 1e-6);
        }

        // Position 1: vision (9.0)
        for (int d = 0; d < hiddenDim; d++) {
            assertEquals(9.0f, merged.getFloat(0, 1, d), 1e-6);
        }

        // Position 2: text (1.0)
        for (int d = 0; d < hiddenDim; d++) {
            assertEquals(1.0f, merged.getFloat(0, 2, d), 1e-6);
        }

        // Position 3: vision (9.0)
        for (int d = 0; d < hiddenDim; d++) {
            assertEquals(9.0f, merged.getFloat(0, 3, d), 1e-6);
        }

        // Position 4: text (1.0)
        for (int d = 0; d < hiddenDim; d++) {
            assertEquals(1.0f, merged.getFloat(0, 4, d), 1e-6);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMergePreservesTextPositions(Nd4jBackend backend) {
        int seqLen = 6;
        int hiddenDim = 3;
        int visionSeqLen = 2;

        // Text embeddings with distinct values per position
        float[] textData = new float[seqLen * hiddenDim];
        for (int i = 0; i < seqLen; i++) {
            for (int d = 0; d < hiddenDim; d++) {
                textData[i * hiddenDim + d] = (i + 1) * 10.0f + d;
            }
        }
        INDArray textEmbeddings = Nd4j.create(textData, new long[]{1, seqLen, hiddenDim});

        INDArray visionEmbeddings = Nd4j.valueArrayOf(new long[]{1, visionSeqLen, hiddenDim}, 99.0f);

        // Image tokens at positions 2 and 4
        int[] tokenIds = {1, 2, IMAGE_TOKEN_ID, 3, IMAGE_TOKEN_ID, 4};

        INDArray merged = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, tokenIds, IMAGE_TOKEN_ID);

        // Non-image positions should preserve original text embeddings
        // Position 0: text value should be 10.0, 11.0, 12.0
        assertEquals(10.0f, merged.getFloat(0, 0, 0), 1e-4);
        assertEquals(11.0f, merged.getFloat(0, 0, 1), 1e-4);
        assertEquals(12.0f, merged.getFloat(0, 0, 2), 1e-4);

        // Position 1: text value should be 20.0, 21.0, 22.0
        assertEquals(20.0f, merged.getFloat(0, 1, 0), 1e-4);

        // Position 2: vision (99.0)
        assertEquals(99.0f, merged.getFloat(0, 2, 0), 1e-4);

        // Position 3: text (40.0, 41.0, 42.0)
        assertEquals(40.0f, merged.getFloat(0, 3, 0), 1e-4);

        // Position 4: vision (99.0)
        assertEquals(99.0f, merged.getFloat(0, 4, 0), 1e-4);

        // Position 5: text (60.0, 61.0, 62.0)
        assertEquals(60.0f, merged.getFloat(0, 5, 0), 1e-4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMergeMismatchedSlots(Nd4jBackend backend) {
        int seqLen = 5;
        int hiddenDim = 2;

        INDArray textEmbeddings = Nd4j.ones(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray visionEmbeddings = Nd4j.valueArrayOf(new long[]{1, 1, hiddenDim}, 9.0f);

        // 3 image slots but only 1 vision token — should handle gracefully
        int[] tokenIds = {IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2};

        INDArray merged = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, tokenIds, IMAGE_TOKEN_ID);

        assertArrayEquals(new long[]{1, seqLen, hiddenDim}, merged.shape());

        // Position 0: filled with vision (9.0)
        assertEquals(9.0f, merged.getFloat(0, 0, 0), 1e-6);

        // Positions 1,2: image tokens but no more vision data → text embedding used
        assertEquals(1.0f, merged.getFloat(0, 1, 0), 1e-6);
        assertEquals(1.0f, merged.getFloat(0, 2, 0), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMergeNoImageTokens(Nd4jBackend backend) {
        int seqLen = 3;
        int hiddenDim = 2;

        INDArray textEmbeddings = Nd4j.ones(DataType.FLOAT, 1, seqLen, hiddenDim).mul(5.0f);
        INDArray visionEmbeddings = Nd4j.valueArrayOf(new long[]{1, 1, hiddenDim}, 9.0f);

        // No image tokens → all text
        int[] tokenIds = {1, 2, 3};

        INDArray merged = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, tokenIds, IMAGE_TOKEN_ID);

        // All positions should be text (5.0)
        for (int pos = 0; pos < seqLen; pos++) {
            for (int d = 0; d < hiddenDim; d++) {
                assertEquals(5.0f, merged.getFloat(0, pos, d), 1e-6);
            }
        }
    }
}
