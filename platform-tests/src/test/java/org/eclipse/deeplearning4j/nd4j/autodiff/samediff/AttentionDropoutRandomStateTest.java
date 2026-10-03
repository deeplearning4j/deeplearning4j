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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

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

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * dot_product_attention_v2 draws its dropout mask from its context's random generator, which SameDiff seeds from
 * Nd4j.getRandom() for every execution that draws random state (ADR 0126). The op draws only with a dropout rate above
 * 0 while training (DeclarableOp::drawsRandomStateFor); its other executions stay capturable. Before, a plan handed the
 * attention slot a generator seeded from the clock, so Nd4j.getRandom().setSeed did not reproduce its output and a
 * gradient check of attention with dropout compared different masks.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class AttentionDropoutRandomStateTest extends BaseNd4jTestWithBackends {

    private static SameDiff attention(boolean training) {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.var("q", Nd4j.rand(DataType.FLOAT, 2, 6, 8).subi(0.5));
        SDVariable k = sd.var("k", Nd4j.rand(DataType.FLOAT, 2, 6, 8).subi(0.5));
        SDVariable v = sd.var("v", Nd4j.rand(DataType.FLOAT, 2, 6, 8).subi(0.5));
        sd.nn().dotProductAttentionV2("out", q, v, k, null, null, 0.0, 0.5, false, training);
        return sd;
    }

    private static INDArray run(SameDiff sd) {
        return sd.output(Collections.emptyMap(), "out").get("out").dup();
    }

    /** The same seed gives the same dropout mask; another seed, or the advanced generator, another mask. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void dropoutFollowsTheThreadGenerator(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = attention(true);

        Nd4j.getRandom().setSeed(7);
        INDArray first = run(sd);
        INDArray advanced = run(sd);
        Nd4j.getRandom().setSeed(7);
        INDArray again = run(sd);
        Nd4j.getRandom().setSeed(8);
        INDArray other = run(sd);

        assertEquals(first, again, "the same seed reproduces the dropout mask");
        assertNotEquals(first, advanced, "the generator moves on after an execution");
        assertNotEquals(first, other, "another seed draws another mask");
    }

    /** Without training no dropout runs: the output does not depend on the generator. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void inferenceDrawsNothing(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = attention(false);

        Nd4j.getRandom().setSeed(7);
        INDArray first = run(sd);
        Nd4j.getRandom().setSeed(8);
        INDArray other = run(sd);
        assertEquals(first, other);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
