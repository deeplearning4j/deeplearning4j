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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class DecoderUtilsTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildCausalMaskPrefill(Nd4jBackend backend) {
        // Multi-token prefill: 4 query tokens, 4 total (no past)
        INDArray mask = ModelIOConfig.buildCausalMask(4, 4);

        assertArrayEquals(new long[]{1, 1, 4, 4}, mask.shape());
        assertEquals(DataType.FLOAT, mask.dataType());

        // Row 0: can attend to position 0 only → [0, FILL, FILL, FILL]
        assertEquals(0.0f, mask.getFloat(0, 0, 0, 0), 1e-6);
        assertEquals(ModelIOConfig.MASK_FILL, mask.getFloat(0, 0, 0, 1), 1e-6);

        // Row 1: can attend to 0,1 → [0, 0, FILL, FILL]
        assertEquals(0.0f, mask.getFloat(0, 0, 1, 0), 1e-6);
        assertEquals(0.0f, mask.getFloat(0, 0, 1, 1), 1e-6);
        assertEquals(ModelIOConfig.MASK_FILL, mask.getFloat(0, 0, 1, 2), 1e-6);

        // Row 3 (last): can attend to all → [0, 0, 0, 0]
        assertEquals(0.0f, mask.getFloat(0, 0, 3, 0), 1e-6);
        assertEquals(0.0f, mask.getFloat(0, 0, 3, 3), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildCausalMaskDecode(Nd4jBackend backend) {
        // Single-token decode: 1 query token, 10 total
        INDArray mask = ModelIOConfig.buildCausalMask(1, 10);

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
        INDArray mask = ModelIOConfig.buildCausalMask(batchSize, 1, 5);

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
        INDArray mask = ModelIOConfig.buildCausalMask(2, 5);

        assertArrayEquals(new long[]{1, 1, 2, 5}, mask.shape());

        // Row 0 (q=0): pastSeqLen=3, can attend to positions 0..3
        assertEquals(0.0f, mask.getFloat(0, 0, 0, 3), 1e-6);
        assertEquals(ModelIOConfig.MASK_FILL, mask.getFloat(0, 0, 0, 4), 1e-6);

        // Row 1 (q=1): pastSeqLen=3, can attend to positions 0..4 (all)
        assertEquals(0.0f, mask.getFloat(0, 0, 1, 4), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReusableDecodePositionIdsAreStableMaterializedLongBuffers(Nd4jBackend backend) {
        ModelIOConfig ioConfig = ModelIOConfig.builder().build();
        List<String> inputNames = new ArrayList<>();
        inputNames.add("position_ids");
        Map<String, INDArray> reusableInputs = new HashMap<>();

        Map<String, INDArray> step0 = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, inputNames, null, null, null,
                679, 1, null, 0, 679,
                true, 0, reusableInputs, true,
                null, null);

        INDArray pos0 = step0.get("position_ids");
        assertNotNull(pos0);
        assertEquals(DataType.LONG, pos0.dataType());
        assertArrayEquals(new long[]{1, 1}, pos0.shape());
        assertFalse(pos0.isView(), "Decode position_ids should be a materialized buffer");
        assertFalse(pos0.wasClosed(), "Decode position_ids must stay live after construction");
        assertEquals(679L, pos0.getLong(0, 0));

        Map<String, INDArray> step1 = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, inputNames, null, null, null,
                680, 1, null, 0, 680,
                true, 0, reusableInputs, true,
                null, null);

        INDArray pos1 = step1.get("position_ids");
        assertSame(pos0, pos1, "Decode position_ids should be reused in-place across steps");
        assertFalse(pos1.isView(), "Reused decode position_ids must remain materialized");
        assertFalse(pos1.wasClosed(), "Reused decode position_ids must remain live");
        assertEquals(680L, pos1.getLong(0, 0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildDecoderInputMapAssociatesInternalInputIdsVariable(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        try {
            SDVariable embeds = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, 4);
            SDVariable inputIds = sd.var("input_ids", Nd4j.zeros(DataType.LONG, 1, 1));
            SDVariable shape = sd.shape("input_ids_shape", inputIds);
            SDVariable tokenValue = sd.castTo("token_value", inputIds, DataType.FLOAT);
            SDVariable passthrough = sd.identity("output", embeds);
            sd.setOutputs("input_ids_shape", "token_value", "output");

            INDArray embedsArr = Nd4j.zeros(DataType.FLOAT, 1, 1, 4);
            INDArray inputIdsArr = Nd4j.createFromArray(new long[]{680L}).reshape(1, 1);
            ModelIOConfig ioConfig = ModelIOConfig.builder()
                    .inputEmbeddingsName("inputs_embeds")
                    .inputIdsName("input_ids")
                    .build();

            Map<String, INDArray> inputMap = DecoderInputBuilder.buildDecoderInputMap(
                    ioConfig, List.of("inputs_embeds"), sd,
                    embedsArr, inputIdsArr,
                    680, 1,
                    null, 0, 0,
                    false, 4,
                    null, false,
                    null, null);

            assertTrue(inputMap.containsKey("input_ids"),
                    "Configured input_ids should be materialized even when omitted from the caller input list");

            Map<String, INDArray> outputs = sd.output(inputMap, "input_ids_shape", "token_value", "output");
            INDArray shapeArr = outputs.get("input_ids_shape");
            INDArray tokenValueArr = outputs.get("token_value");
            assertNotNull(shapeArr);
            assertNotNull(tokenValueArr);
            assertArrayEquals(new long[]{2}, shapeArr.shape());
            assertArrayEquals(new long[]{1, 1}, shapeArr.toLongVector(),
                    "Internal input_ids shape should come from the associated per-step array");
            assertEquals(680.0f, tokenValueArr.getFloat(0, 0), 1e-6f,
                    "Internal input_ids value should come from the associated per-step array");
        } finally {
            sd.close();
        }
    }
}
