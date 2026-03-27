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

package org.eclipse.deeplearning4j.nd4j.linalg.shape;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for the permute op's rank mismatch adaptation logic.
 *
 * When a permute op receives input with more dimensions than its permutation vector
 * (e.g., rank-5 input with a 4-element permutation), the op detects leading size-1
 * dimensions added by expand_dims, keeps them as identity-mapped, and shifts the original
 * permutation indices. For example: input rank-5, perm [0,2,3,1] -> adapted perm [0,1,3,4,2].
 *
 * Tests use the Permute op class via Nd4j.exec() to exercise the Java-level adaptation
 * in Permute.initializeOutputs(), plus SameDiff for the matching-rank regression check.
 */
@NativeTag
public class TestPermuteRankMismatchAdaptation extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Helper: execute permute using the Permute op class directly.
     */
    private INDArray execPermute(INDArray input, long... permDims) {
        Permute op = new Permute(input, permDims);
        INDArray[] results = Nd4j.exec(op);
        return results[0];
    }

    /**
     * Helper: execute a SameDiff permute with the given input and permutation dims.
     */
    private INDArray sdPermute(INDArray input, long... permDims) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("input", input);
        SDVariable out = sd.permute("output", in, permDims);
        Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
        return result.get("output");
    }

    /**
     * Test 1: Vision encoder case — rank-5 input with rank-4 permutation.
     * Input [1,1,3,512,512] with perm [0,2,3,1] -> adapted [0,1,3,4,2] -> output [1,1,512,512,3]
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRank5WithRank4Perm_visionEncoder(Nd4jBackend backend) {
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 512, 512);
        INDArray result = execPermute(input, 0, 2, 3, 1);
        assertArrayEquals(new long[]{1, 1, 512, 512, 3}, result.shape(),
                "Vision encoder case: output shape mismatch");
    }

    /**
     * Test 2: Normal case — rank-4 input with matching rank-4 permutation (regression check).
     * Uses both Permute op and SameDiff permute to verify both paths.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRank4WithRank4Perm_normalCase(Nd4jBackend backend) {
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 64, 64);

        // Via Permute op
        INDArray result = execPermute(input, 0, 2, 3, 1);
        assertArrayEquals(new long[]{1, 64, 64, 3}, result.shape(),
                "Normal rank-4 case: Permute op output shape mismatch");

        // Via SameDiff
        INDArray sdResult = sdPermute(input, 0, 2, 3, 1);
        assertArrayEquals(new long[]{1, 64, 64, 3}, sdResult.shape(),
                "Normal rank-4 case: SameDiff output shape mismatch");
    }

    /**
     * Test 3: Two extra size-1 dims — rank-6 input with rank-4 permutation.
     * Input [1,1,1,3,8,8] with perm [0,2,3,1] -> adapted [0,1,2,4,5,3] -> output [1,1,1,8,8,3]
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRank6WithRank4Perm_twoExtraDims(Nd4jBackend backend) {
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 1, 3, 8, 8);
        INDArray result = execPermute(input, 0, 2, 3, 1);
        assertArrayEquals(new long[]{1, 1, 1, 8, 8, 3}, result.shape(),
                "Two extra dims: output shape mismatch");
    }

    /**
     * Test 4: Non-leading size-1 dims — rank-5 input but first dim is NOT 1.
     * Input [2,1,3,8,8] with perm [0,2,3,1]. Only 1 leading size-1 dim needed
     * but first dim is 2, so adaptation falls back to identity.
     * Output shape should equal input shape [2,1,3,8,8].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRank5WithRank4Perm_nonLeadingOnes(Nd4jBackend backend) {
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 1, 3, 8, 8);
        INDArray result = execPermute(input, 0, 2, 3, 1);
        assertArrayEquals(new long[]{2, 1, 3, 8, 8}, result.shape(),
                "Non-leading ones: should fall back to identity");
    }

    /**
     * Test 5: Permutation larger than input rank.
     * Input [3,8,8] (rank-3) with perm [0,2,3,1]. Valid indices < 3 are [0,2,1]
     * which has 3 elements matching rank 3. Output shape: permute [3,8,8] by [0,2,1] = [3,8,8].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRank3WithRank4Perm_permLargerThanInput(Nd4jBackend backend) {
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 8, 8);
        INDArray result = execPermute(input, 0, 2, 3, 1);
        // Filtered valid indices: [0, 2, 1] (index 3 dropped). Permuting [3,8,8] by [0,2,1] = [3,8,8]
        assertArrayEquals(new long[]{3, 8, 8}, result.shape(),
                "Perm larger than input: output shape mismatch");
    }

    /**
     * Test 6: Value correctness for rank-5 vision encoder case with known data.
     * Input [1,1,2,3,4] with linspace 0..23, perm [0,2,3,1].
     * Adapted perm: [0,1,3,4,2] -> output shape [1,1,3,4,2].
     * Verify output(0,0,i,j,k) == input(0,0,k,i,j).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testValueCorrectness_rank5VisionEncoder(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(0, 23, 24, DataType.FLOAT).reshape(1, 1, 2, 3, 4);
        INDArray output = execPermute(input, 0, 2, 3, 1);

        assertArrayEquals(new long[]{1, 1, 3, 4, 2}, output.shape(),
                "Value correctness: output shape mismatch");

        // Adapted perm is [0,1,3,4,2], so output(b0,b1,i,j,k) = input(b0,b1,k,i,j)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 2; k++) {
                    float expected = input.getFloat(0, 0, k, i, j);
                    float actual = output.getFloat(0, 0, i, j, k);
                    assertEquals(expected, actual, 1e-5f,
                            String.format("Mismatch at output(0,0,%d,%d,%d): expected input(0,0,%d,%d,%d)=%.1f got %.1f",
                                    i, j, k, k, i, j, expected, actual));
                }
            }
        }
    }

    /**
     * Test 7: Identity permutation with extra dims.
     * Input [1,1,3,8,8] with perm [0,1,2] (identity for rank-3).
     * Adapted: [0,1,2,3,4] (identity for rank-5). Output shape should equal input shape.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIdentityPermWithExtraDims(Nd4jBackend backend) {
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 8, 8);
        INDArray result = execPermute(input, 0, 1, 2);
        assertArrayEquals(new long[]{1, 1, 3, 8, 8}, result.shape(),
                "Identity perm with extra dims: output shape should equal input");
        assertEquals(input, result, "Identity permutation should not change values");
    }

    /**
     * Test 8: Shape consistency — verify the Permute op produces correct adapted shapes
     * for various rank mismatch scenarios.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeFunctionConsistency(Nd4jBackend backend) {
        // Case A: rank-5 with rank-4 perm (vision encoder)
        verifyShapeConsistency(new long[]{1, 1, 3, 16, 16}, new long[]{0, 2, 3, 1}, new long[]{1, 1, 16, 16, 3});

        // Case B: normal rank-4 with rank-4 perm
        verifyShapeConsistency(new long[]{1, 3, 16, 16}, new long[]{0, 2, 3, 1}, new long[]{1, 16, 16, 3});

        // Case C: rank-6 with rank-4 perm (two extra dims)
        verifyShapeConsistency(new long[]{1, 1, 1, 3, 8, 8}, new long[]{0, 2, 3, 1}, new long[]{1, 1, 1, 8, 8, 3});

        // Case D: identity perm with extra dims
        verifyShapeConsistency(new long[]{1, 1, 3, 8, 8}, new long[]{0, 1, 2}, new long[]{1, 1, 3, 8, 8});
    }

    private void verifyShapeConsistency(long[] inputShape, long[] perm, long[] expectedShape) {
        INDArray input = Nd4j.rand(DataType.FLOAT, inputShape);
        INDArray result = execPermute(input, perm);

        assertArrayEquals(expectedShape, result.shape(),
                String.format("Shape consistency: input %s perm %s -> expected %s got %s",
                        java.util.Arrays.toString(inputShape),
                        java.util.Arrays.toString(perm),
                        java.util.Arrays.toString(expectedShape),
                        java.util.Arrays.toString(result.shape())));
    }
}
