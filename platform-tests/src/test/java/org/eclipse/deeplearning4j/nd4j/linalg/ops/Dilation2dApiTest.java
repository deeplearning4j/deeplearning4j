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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Dilation2D;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.factory.ops.NDCNN;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * dilation2d, the grayscale dilation of TensorFlow: out[b, y, x, c] is the largest of in[b, y * sH + i * rH - padTop,
 * x * sW + j * rW - padLeft, c] + weights[i, j, c] over the window, the taps outside the image skipped. The SameDiff and
 * INDArray APIs take the strides and rates as (height, width) pairs, which the op class refused (it wanted four
 * values), and its native op typed the output by the weights while its helpers read and wrote everything as the input's
 * type.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class Dilation2dApiTest extends BaseNd4jTestWithBackends {

    // strides (height, width), rates (height, width), same mode
    private static final Object[][] CASES = {
            {new int[]{1, 1}, new int[]{1, 1}, false},
            {new int[]{1, 1}, new int[]{1, 1}, true},
            {new int[]{2, 1}, new int[]{1, 2}, true},
            {new int[]{1, 2}, new int[]{2, 1}, false},
            {new int[]{2, 2}, new int[]{2, 2}, true},
    };

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    private static INDArray image(DataType type) {
        Nd4j.getRandom().setSeed(5);
        return Nd4j.rand(DataType.DOUBLE, 2, 6, 7, 3).castTo(type);
    }

    private static INDArray weights(DataType type) {
        Nd4j.getRandom().setSeed(6);
        return Nd4j.rand(DataType.DOUBLE, 3, 2, 3).castTo(type);
    }

    /** TensorFlow's dilation2d, from its definition; the output's shape is [batch, oH, oW, channels]. */
    private static INDArray reference(INDArray in, INDArray w, int[] strides, int[] rates, boolean same) {
        int batch = (int) in.size(0), iH = (int) in.size(1), iW = (int) in.size(2), channels = (int) in.size(3);
        int kH = (int) w.size(0), kW = (int) w.size(1);
        int kHeff = kH + (kH - 1) * (rates[0] - 1);
        int kWeff = kW + (kW - 1) * (rates[1] - 1);
        int oH, oW, padTop, padLeft;
        if (same) {
            oH = (iH + strides[0] - 1) / strides[0];
            oW = (iW + strides[1] - 1) / strides[1];
            padTop = Math.max(0, (oH - 1) * strides[0] + kHeff - iH) / 2;
            padLeft = Math.max(0, (oW - 1) * strides[1] + kWeff - iW) / 2;
        } else {
            oH = (iH - kHeff + strides[0]) / strides[0];
            oW = (iW - kWeff + strides[1]) / strides[1];
            padTop = 0;
            padLeft = 0;
        }
        INDArray out = Nd4j.zeros(DataType.DOUBLE, batch, oH, oW, channels);
        for (int b = 0; b < batch; b++)
            for (int y = 0; y < oH; y++)
                for (int x = 0; x < oW; x++)
                    for (int c = 0; c < channels; c++) {
                        double max = -Double.MAX_VALUE;
                        for (int i = 0; i < kH; i++) {
                            int row = y * strides[0] - padTop + i * rates[0];
                            if (row < 0 || row >= iH)
                                continue;
                            for (int j = 0; j < kW; j++) {
                                int col = x * strides[1] - padLeft + j * rates[1];
                                if (col < 0 || col >= iW)
                                    continue;
                                max = Math.max(max, in.getDouble(b, row, col, c) + w.getDouble(i, j, c));
                            }
                        }
                        out.putScalar(new long[]{b, y, x, c}, max);
                    }
        return out;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void apisTakeStridesAndRatesAsPairs(Nd4jBackend backend) {
        NDCNN cnn = new NDCNN();
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray in = image(type);
            INDArray w = weights(type);
            for (Object[] c : CASES) {
                int[] strides = (int[]) c[0];
                int[] rates = (int[]) c[1];
                boolean same = (Boolean) c[2];
                INDArray expected = reference(in, w, strides, rates, same);
                String where = type + " strides " + Arrays.toString(strides) + ", rates " + Arrays.toString(rates)
                        + (same ? ", same" : ", valid");
                double tolerance = type == DataType.DOUBLE ? 1e-12 : 1e-5;

                // the INDArray API
                INDArray out = cnn.dilation2D(in, w, strides, rates, same);
                assertEquals(type, out.dataType(), where);
                assertArrayEquals(expected.shape(), out.shape(), where);
                assertArrayEquals(logical(expected), logical(out), tolerance, "INDArray API, " + where);

                // the SameDiff API
                SameDiff sd = SameDiff.create();
                SDVariable result = sd.cnn().dilation2D(sd.constant("in", in), sd.constant("w", w), strides, rates, same);
                assertArrayEquals(logical(expected), logical(result.eval()), tolerance, "SameDiff API, " + where);

                // and the op class, with the four values of the NHWC layout
                INDArray[] fourValued = Nd4j.exec(new Dilation2D(in, w, new int[]{1, strides[0], strides[1], 1},
                        new int[]{1, rates[0], rates[1], 1}, same));
                assertArrayEquals(logical(expected), logical(fourValued[0]), tolerance, "four values, " + where);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stridesAndRatesOfAnotherLengthAreRefused(Nd4jBackend backend) {
        INDArray in = image(DataType.FLOAT);
        INDArray w = weights(DataType.FLOAT);
        assertThrows(IllegalArgumentException.class,
                () -> new Dilation2D(in, w, new int[]{1, 1, 1}, new int[]{1, 1}, false));
        assertThrows(IllegalArgumentException.class,
                () -> new Dilation2D(in, w, new int[]{1, 1}, new int[]{1}, false));
    }

    /** The weights are read as the input's type, so they must be of it: a different type is an error, not garbage. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void weightsOfAnotherTypeAreRefused(Nd4jBackend backend) {
        INDArray in = image(DataType.FLOAT);
        INDArray w = weights(DataType.DOUBLE);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(DynamicCustomOp.builder("dilation2d").addInputs(in, w)
                .addIntegerArguments(0, 1, 1, 1, 1, 1, 1, 1, 1).build()));
    }
}
