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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class DecoderUtilsTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildCausalMaskPrefill(Nd4jBackend backend) {
        // Multi-token prefill: 4 query tokens, 4 total (no past)
        INDArray mask = DecoderUtils.buildCausalMask(4, 4);

        assertArrayEquals(new long[]{1, 1, 4, 4}, mask.shape());
        assertEquals(DataType.FLOAT, mask.dataType());

        // Row 0: can attend to position 0 only → [0, FILL, FILL, FILL]
        assertEquals(0.0f, mask.getFloat(0, 0, 0, 0), 1e-6);
        assertEquals(DecoderUtils.MASK_FILL, mask.getFloat(0, 0, 0, 1), 1e-6);

        // Row 1: can attend to 0,1 → [0, 0, FILL, FILL]
        assertEquals(0.0f, mask.getFloat(0, 0, 1, 0), 1e-6);
        assertEquals(0.0f, mask.getFloat(0, 0, 1, 1), 1e-6);
        assertEquals(DecoderUtils.MASK_FILL, mask.getFloat(0, 0, 1, 2), 1e-6);

        // Row 3 (last): can attend to all → [0, 0, 0, 0]
        assertEquals(0.0f, mask.getFloat(0, 0, 3, 0), 1e-6);
        assertEquals(0.0f, mask.getFloat(0, 0, 3, 3), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildCausalMaskDecode(Nd4jBackend backend) {
        // Single-token decode: 1 query token, 10 total
        INDArray mask = DecoderUtils.buildCausalMask(1, 10);

        assertArrayEquals(new long[]{1, 1, 1, 10}, mask.shape());
        assertEquals(DataType.FLOAT, mask.dataType());

        // All zeros — single token attends to all past
        for (int k = 0; k < 10; k++) {
            assertEquals(0.0f, mask.getFloat(0, 0, 0, k), 1e-6);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildCausalMaskBatched(Nd4jBackend backend) {
        int batchSize = 3;
        INDArray mask = DecoderUtils.buildCausalMask(batchSize, 1, 5);

        // batch=3, heads=1, q=1, k=5
        assertArrayEquals(new long[]{3, 1, 1, 5}, mask.shape());

        // All zeros for decode step
        for (int b = 0; b < batchSize; b++) {
            for (int k = 0; k < 5; k++) {
                assertEquals(0.0f, mask.getFloat(b, 0, 0, k), 1e-6);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCausalMaskWithPast(Nd4jBackend backend) {
        // 2 query tokens, 5 total (3 past + 2 current)
        INDArray mask = DecoderUtils.buildCausalMask(2, 5);

        assertArrayEquals(new long[]{1, 1, 2, 5}, mask.shape());

        // Row 0 (q=0): pastSeqLen=3, can attend to positions 0..3
        assertEquals(0.0f, mask.getFloat(0, 0, 0, 3), 1e-6);
        assertEquals(DecoderUtils.MASK_FILL, mask.getFloat(0, 0, 0, 4), 1e-6);

        // Row 1 (q=1): pastSeqLen=3, can attend to positions 0..4 (all)
        assertEquals(0.0f, mask.getFloat(0, 0, 1, 4), 1e-6);
    }
}
