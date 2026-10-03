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

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * gruCell_bp is the gradient of gruCell: with gradients Gr, Gu, Gc and Gh wrt gruCell's outputs r, u, c and
 * h = u * hLast + (1 - u) * c, its outputs are the derivatives of L = sum(Gr * r + Gu * u + Gc * c + Gh * h) wrt x,
 * hLast, the gate and cell weights and the biases, checked against central differences. The backprop ignored the
 * paths from h into u and c and from c into r, ordered the product with r on the path from hLast through c the wrong
 * way, and read freed copies of its gate activations.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class GruCellBackpropTest extends BaseNd4jTestWithBackends {

    private static INDArray[] gruCell(INDArray[] in) {
        return Nd4j.exec(DynamicCustomOp.builder("gruCell").addInputs(in).build());
    }

    /** L = sum over the four outputs of the output times its gradient. */
    private static double loss(INDArray[] in, INDArray[] grads) {
        INDArray[] out = gruCell(in);
        double l = 0;
        for (int k = 0; k < 4; k++)
            l += out[k].mul(grads[k]).sumNumber().doubleValue();
        return l;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backpropIsTheGradientOfTheCell(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(47);
        int bS = 2, iS = 3, nU = 4;
        String[] names = {"x", "hLast", "W", "Wc", "b", "bc"};
        INDArray[] in = {
                Nd4j.rand(DataType.DOUBLE, bS, iS).subi(0.5),
                Nd4j.rand(DataType.DOUBLE, bS, nU).subi(0.5),
                Nd4j.rand(DataType.DOUBLE, iS + nU, 2 * nU).subi(0.5),
                Nd4j.rand(DataType.DOUBLE, iS + nU, nU).subi(0.5),
                Nd4j.rand(DataType.DOUBLE, 2 * nU).subi(0.5),
                Nd4j.rand(DataType.DOUBLE, nU).subi(0.5)};
        // gradients wrt r, u, c and h
        INDArray[] grads = new INDArray[4];
        for (int k = 0; k < 4; k++)
            grads[k] = Nd4j.rand(DataType.DOUBLE, bS, nU).subi(0.5);

        INDArray[] bpIn = new INDArray[10];
        System.arraycopy(in, 0, bpIn, 0, 6);
        System.arraycopy(grads, 0, bpIn, 6, 4);
        INDArray[] analytic = Nd4j.exec(DynamicCustomOp.builder("gruCell_bp").addInputs(bpIn).build());

        double eps = 1e-6;
        List<String> failures = new ArrayList<>();
        for (int p = 0; p < 6; p++) {
            INDArray param = in[p];
            for (int i = 0; i < param.length(); i++) {
                double orig = param.getDouble(i);
                param.putScalar(i, orig + eps);
                double plus = loss(in, grads);
                param.putScalar(i, orig - eps);
                double minus = loss(in, grads);
                param.putScalar(i, orig);
                double numeric = (plus - minus) / (2 * eps);
                double a = analytic[p].getDouble(i);
                if (Math.abs(a - numeric) > 1e-6 * Math.max(1, Math.abs(numeric)))
                    failures.add("d L / d " + names[p] + "[" + i + "]: " + a + " vs numeric " + numeric);
            }
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }
}
