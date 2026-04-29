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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression tests for the Java-side iArg packing contract used by the native
 * autoregressive decode op.
 */
public class TestAutoregressiveDecodeIArgs {

    private static AutoregressiveDecode newOp(int attnMaskReformatExtIdx, Set<Integer> stopIds) {
        INDArray prefillEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 4);
        INDArray embeddingTable = Nd4j.zeros(DataType.FLOAT, 8, 4);
        INDArray inputIds = Nd4j.zeros(DataType.INT64, 1, 1);
        INDArray attentionMask = Nd4j.zeros(DataType.INT64, 1, 6);
        INDArray positionIds = Nd4j.zeros(DataType.INT64, 1, 1);
        INDArray[] staticKvBuffers = new INDArray[]{
                Nd4j.zeros(DataType.FLOAT, 1, 2, 6, 4),
                Nd4j.zeros(DataType.FLOAT, 1, 2, 6, 4)
        };

        return new AutoregressiveDecode(
                prefillEmbeddings,
                embeddingTable,
                inputIds,
                attentionMask,
                positionIds,
                staticKvBuffers,
                null,
                null,
                10,
                5,
                1,
                2,
                3,
                4,
                5,
                6,
                attnMaskReformatExtIdx,
                -1,  // cachePositionExtIdx — disabled for this test
                new int[]{101, 102},
                new int[]{201, 202},
                7,
                99,
                1,
                11,
                0.0,
                0,
                0.0,
                stopIds);
    }

    @Test
    @DisplayName("AutoregressiveDecode iArgs omit attn_mask_reformat slot when absent")
    public void testIArgsWithoutAttnMaskReformat() {
        AutoregressiveDecode op = newOp(-1, Set.of(777));
        long[] iArgs = op.iArgs();

        assertEquals(7L, iArgs[4], "optionalMask should include mask, posIds, and KV only");
        assertArrayEquals(new long[]{101, 102, 201, 202, 777},
                java.util.Arrays.copyOfRange(iArgs, 17, 22),
                "KV indices must start at iArg[17] when attn_mask_reformat is absent");
    }

    @Test
    @DisplayName("AutoregressiveDecode iArgs include attn_mask_reformat slot when present")
    public void testIArgsWithAttnMaskReformat() {
        AutoregressiveDecode op = newOp(77, Set.of());
        long[] iArgs = op.iArgs();

        assertEquals(15L, iArgs[4], "optionalMask should include attn_mask_reformat when present");
        assertArrayEquals(new long[]{77, 101, 102, 201, 202},
                java.util.Arrays.copyOfRange(iArgs, 17, 22),
                "attn_mask_reformat index must occupy iArg[17] when present");
    }
}
