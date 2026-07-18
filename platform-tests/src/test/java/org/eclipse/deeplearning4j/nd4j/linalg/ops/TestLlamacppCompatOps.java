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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the llama.cpp-compat op-name surface: the op names the
 * removed platform/llamacpp helpers used to be the sole provider of, now served
 * natively via DECLARE_SYN registrations (get_rows, gqa_attention,
 * quantized_mul_mat, dequantize, alibi_position_bias) and thin native adapters
 * (scale, add_inplace, get_rows_bp, paged_attention, moe_expert_ffn).
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestLlamacppCompatOps 2>&1 | tee /tmp/test-llamacpp-compat.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestLlamacppCompatOps {

    private static INDArray[] execByName(String opName, INDArray[] inputs, double[] tArgs, long[] iArgs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(opName).addInputs(inputs);
        if (tArgs != null && tArgs.length > 0) builder.addFloatingPointArguments(toBoxed(tArgs));
        if (iArgs != null && iArgs.length > 0) builder.addIntegerArguments(iArgs);
        return Nd4j.exec(builder.build());
    }

    private static Double[] toBoxed(double[] arr) {
        Double[] out = new Double[arr.length];
        for (int i = 0; i < arr.length; i++) out[i] = arr[i];
        return out;
    }

    @Test
    public void testAllCompatNamesAreRegistered() {
        // The production failure mode of these names was "Could not find descriptor for op: X".
        Map<String, CustomOpDescriptor> ops = Nd4j.getExecutioner().getCustomOperations();
        for (String name : new String[]{"get_rows", "gqa_attention", "quantized_mul_mat", "dequantize",
                "alibi_position_bias", "scale", "add_inplace", "get_rows_bp", "paged_attention",
                "moe_expert_ffn"}) {
            assertTrue(ops.containsKey(name), "op descriptor missing for llama.cpp-compat name: " + name);
        }
    }

    @Test
    public void testGetRowsEqualsGather() {
        Nd4j.getRandom().setSeed(42);
        INDArray weights = Nd4j.rand(DataType.FLOAT, 10, 6);
        INDArray indices = Nd4j.createFromArray(3L, 0L, 7L, 3L);

        INDArray viaSyn = execByName("get_rows", new INDArray[]{weights, indices}, null, null)[0];
        INDArray viaGather = execByName("gather", new INDArray[]{weights, indices}, null, new long[]{0})[0];

        assertArrayEquals(new long[]{4, 6}, viaSyn.shape());
        assertEquals(viaGather, viaSyn);
    }

    @Test
    public void testGqaAttentionEqualsGroupedQueryAttention() {
        Nd4j.getRandom().setSeed(43);
        INDArray q = Nd4j.rand(DataType.FLOAT, 1, 4, 2, 8);
        INDArray k = Nd4j.rand(DataType.FLOAT, 1, 4, 2, 8);
        INDArray v = Nd4j.rand(DataType.FLOAT, 1, 4, 2, 8);

        INDArray viaSyn = execByName("gqa_attention", new INDArray[]{q, k, v}, null, null)[0];
        INDArray viaNative = execByName("grouped_query_attention", new INDArray[]{q, k, v}, null, null)[0];

        assertArrayEquals(q.shape(), viaSyn.shape());
        assertEquals(viaNative, viaSyn);
    }

    @Test
    public void testAlibiPositionBiasEqualsApplyAlibi() {
        Nd4j.getRandom().setSeed(44);
        INDArray scores = Nd4j.rand(DataType.FLOAT, 1, 2, 3, 3);

        INDArray viaSyn = execByName("alibi_position_bias", new INDArray[]{scores}, null, null)[0];
        INDArray viaNative = execByName("apply_alibi", new INDArray[]{scores}, null, null)[0];

        assertEquals(viaNative, viaSyn);
    }

    @Test
    public void testScale() {
        INDArray x = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        INDArray out = execByName("scale", new INDArray[]{x}, new double[]{2.5}, null)[0];
        assertEquals(x.mul(2.5), out);

        // default scale = 1.0
        INDArray identity = execByName("scale", new INDArray[]{x}, null, null)[0];
        assertEquals(x, identity);
    }

    @Test
    public void testAddInplace() {
        INDArray acc = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(2, 3);
        INDArray in = Nd4j.linspace(10, 60, 6, DataType.FLOAT).reshape(2, 3);
        INDArray out = execByName("add_inplace", new INDArray[]{acc, in}, null, null)[0];
        assertEquals(acc.add(in), out);
    }

    @Test
    public void testGetRowsBpScatterAdd() {
        // gradWeights[indices[i], :] += gradOutput[i, :], duplicate index accumulates
        INDArray gradOut = Nd4j.createFromArray(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        INDArray indices = Nd4j.createFromArray(2L, 0L, 2L);
        int numRows = 4;

        INDArray gradWeights = execByName("get_rows_bp", new INDArray[]{gradOut, indices},
                null, new long[]{numRows})[0];

        INDArray expected = Nd4j.zeros(DataType.FLOAT, numRows, 2);
        expected.putRow(0, Nd4j.createFromArray(3f, 4f));
        expected.putRow(2, Nd4j.createFromArray(6f, 8f));  // rows 0 and 2 of gradOut accumulated

        assertArrayEquals(new long[]{numRows, 2}, gradWeights.shape());
        assertEquals(expected, gradWeights);
    }

    @Test
    public void testPagedAttentionDelegatesToPagedAttentionForward() {
        Nd4j.getRandom().setSeed(45);
        int batch = 1, heads = 2, kvHeads = 2, headDim = 8, blockSize = 4, numBlocks = 2;
        INDArray q = Nd4j.rand(DataType.FLOAT, batch, 1, heads, headDim);
        INDArray kPool = Nd4j.rand(DataType.FLOAT, numBlocks, blockSize, kvHeads, headDim);
        INDArray vPool = Nd4j.rand(DataType.FLOAT, numBlocks, blockSize, kvHeads, headDim);
        INDArray pageTables = Nd4j.createFromArray(new int[][]{{0, 1}});
        INDArray contextLens = Nd4j.createFromArray(new int[]{6});

        INDArray viaCompat = execByName("paged_attention",
                new INDArray[]{q, kPool, vPool, pageTables, contextLens}, null, new long[]{blockSize})[0];
        INDArray viaNative = execByName("paged_attention_forward",
                new INDArray[]{q, kPool, vPool, pageTables, contextLens},
                new double[]{0.0}, new long[]{blockSize, heads, kvHeads, headDim})[0];

        assertEquals(viaNative, viaCompat);
    }

    @Test
    public void testMoeExpertFfnAgainstJavaReference() {
        Nd4j.getRandom().setSeed(46);
        int tokens = 3, hidden = 4, experts = 3, expertOut = 5, topK = 2;
        INDArray input = Nd4j.rand(DataType.FLOAT, tokens, hidden).subi(0.5);
        INDArray expertWeights = Nd4j.rand(DataType.FLOAT, experts, hidden, expertOut).subi(0.5);
        INDArray routing = Nd4j.createFromArray(new float[][]{{0.7f, 0.3f}, {0.5f, 0.5f}, {0.9f, 0.1f}});
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 2}, {1, 0}, {2, 1}});

        INDArray out = execByName("moe_expert_ffn",
                new INDArray[]{input, expertWeights, routing, indices}, null, new long[]{experts})[0];

        assertArrayEquals(new long[]{tokens, expertOut}, out.shape());
        for (int t = 0; t < tokens; t++) {
            for (int d = 0; d < expertOut; d++) {
                double expected = 0;
                for (int k = 0; k < topK; k++) {
                    long e = indices.getLong(t, k);
                    double dot = 0;
                    for (int h = 0; h < hidden; h++)
                        dot += input.getDouble(t, h) * expertWeights.getDouble(e, h, d);
                    expected += routing.getDouble(t, k) * dot;
                }
                assertEquals(expected, out.getDouble(t, d), 1e-4,
                        "moe_expert_ffn mismatch at [" + t + "," + d + "]");
            }
        }
    }
}
