/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.sparse;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression test: gradcheck a Gram matrix x·xᵀ twice, as two test methods in one JVM. The second
 * method's {@code Nd4j.randn} used to throw "Ptr data buffer was released!".
 *
 * <p>Root cause (not the op or the Graph namespace): the {@code @AfterEach} teardown in
 * {@code BaseND4JTest.reclaimGpuMemory()} runs {@code DeallocatorService.forceFlushAll()}, which
 * frees the native memory of every refMap buffer — including the {@code ConstantHandler}'s
 * strongly-cached constants such as the GaussianDistribution {@code [mean,stddev]} extra-args
 * buffer that {@code Nd4j.randn} fetches. The constant cache kept the dead DataBuffer wrapper, so
 * the next test got a buffer over freed memory. Fixed by pairing
 * {@code Nd4j.getConstantHandler().purgeConstants()} with the force-flush in that teardown.
 */
@Tag(TagNames.SAMEDIFF)
public class SelfOpReproTest extends BaseNd4jTestWithBackends {

    private static void gramGradCheck(long seed) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, 4, 3));
            SDVariable gram = sd.mmul(x, x, false, true, false);   // x · xᵀ  -- x is BOTH operands
            sd.sum("loss", gram);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "gram grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGramA(Nd4jBackend backend) { gramGradCheck(1L); }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGramB(Nd4jBackend backend) { gramGradCheck(2L); }
}
