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
 * Gradient checks for the Discrete Fourier Transform op. The DFT is a linear transform, so its
 * gradient is the adjoint (inverse) transform with an N / 1/N scaling; this validates that
 * DFT.doDiff is correct on both backends, which is what unblocks HolE's circular-correlation scorer.
 * Complex tensors use a trailing dimension of size 2 for [real, imag].
 */
@Tag(TagNames.SAMEDIFF)
public class DftGradientTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testForwardDftGrad(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(1L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, 2, 5, 2)); // [batch, N, (re,im)]
            SDVariable X = sd.signal().dft(x, 1, false, false);
            sd.sum("loss", sd.math().square(X));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "forward DFT grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInverseDftGrad(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(2L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, 2, 5, 2));
            SDVariable X = sd.signal().dft(x, 1, true, false);
            sd.sum("loss", sd.math().square(X));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "inverse DFT grad check failed");
        } finally {
            sd.close();
        }
    }

    /** round-trip IDFT(DFT(x)) == x, sanity that the convention/scaling is self-consistent. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDftRoundTrip(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(3L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, 2, 6, 2));
            SDVariable rt = sd.signal().dft(sd.signal().dft(x, 1, false, false), 1, true, false);
            double maxDiff = Nd4j.math().abs(rt.eval().sub(x.eval())).maxNumber().doubleValue();
            System.err.println("DFT round-trip max |IDFT(DFT(x)) - x| = " + maxDiff);
            assertTrue(maxDiff < 1e-6, "IDFT(DFT(x)) != x (maxDiff=" + maxDiff + ")");
        } finally {
            sd.close();
        }
    }
}
