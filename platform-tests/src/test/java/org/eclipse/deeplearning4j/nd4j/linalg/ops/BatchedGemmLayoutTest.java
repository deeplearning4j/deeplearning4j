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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * batched_gemm multiplies alpha op(A_i) op(B_i) for each pair, whatever order the matrices lie in. The CPU helper
 * read every matrix with BLAS's column-major leading dimensions, so a 'c' matrix (SameDiff's default) was read as its
 * transpose; and its 16-bit fallback summed in the 16-bit type.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class BatchedGemmLayoutTest extends BaseNd4jTestWithBackends {

    private static final int M = 4;
    private static final int N = 3;
    private static final int K = 5;
    private static final double ALPHA = 1.5;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyLayoutTransposeAndTypeMultiplies(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.BFLOAT16}) {
            for (char aOrder : new char[]{'c', 'f'}) {
                for (char bOrder : new char[]{'c', 'f'}) {
                    for (boolean transA : new boolean[]{false, true}) {
                        for (boolean transB : new boolean[]{false, true}) {
                            check(type, aOrder, bOrder, transA, transB);
                        }
                    }
                }
            }
        }
    }

    private static void check(DataType type, char aOrder, char bOrder, boolean transA, boolean transB) {
        String label = type + " A " + aOrder + (transA ? "^T" : "") + " B " + bOrder + (transB ? "^T" : "");
        Nd4j.getRandom().setSeed(11);
        INDArray[] a = new INDArray[2];
        INDArray[] b = new INDArray[2];
        for (int i = 0; i < 2; i++) {
            a[i] = Nd4j.rand(DataType.DOUBLE, transA ? K : M, transA ? M : K).subi(0.5).castTo(type).dup(aOrder);
            b[i] = Nd4j.rand(DataType.DOUBLE, transB ? N : K, transB ? K : N).subi(0.5).castTo(type).dup(bOrder);
        }
        INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("batched_gemm")
                .addInputs(Nd4j.scalar(type, ALPHA), Nd4j.scalar(type, 0.0), a[0], a[1], b[0], b[1])
                .addIntegerArguments(transA ? 1 : 0, transB ? 1 : 0)
                .build());
        assertEquals(2, out.length, label);
        double tolerance = type == DataType.DOUBLE ? 1e-12 : type == DataType.FLOAT ? 1e-5
                : type == DataType.HALF ? 1e-2 : 4e-2;
        for (int i = 0; i < 2; i++) {
            INDArray opA = a[i].castTo(DataType.DOUBLE);
            INDArray opB = b[i].castTo(DataType.DOUBLE);
            INDArray expected = (transA ? opA.transpose() : opA).mmul(transB ? opB.transpose() : opB).muli(ALPHA);
            assertEquals(type, out[i].dataType(), label);
            assertArrayEquals(new long[]{M, N}, out[i].shape(), label);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    double want = expected.getDouble(m, n);
                    assertEquals(want, out[i].getDouble(m, n), tolerance * (1 + Math.abs(want)),
                            label + " batch " + i + " [" + m + ", " + n + "]");
                }
            }
        }
    }
}
