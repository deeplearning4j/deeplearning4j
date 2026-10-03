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
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * rgb_to_hsv and hsv_to_rgb convert every pixel of a large image as they convert it alone. The CPU fast path stepped
 * through the elements by 3 in chunks Threads::parallel_for split by element count, not on the pixel grid: from about
 * 2048 pixels, threads after the first mixed channels of neighbouring pixels and the last read and wrote past the end.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class ColorSpaceConversionTest extends BaseNd4jTestWithBackends {

    private static INDArray convert(String op, INDArray in) {
        return Nd4j.exec(DynamicCustomOp.builder(op).addInputs(in).build())[0];
    }

    /** TensorFlow's RGB to HSV (adjust_hue.h), hue in [0, 1). */
    private static float[] rgbToHsv(float r, float g, float b) {
        float max = Math.max(r, Math.max(g, b));
        float min = Math.min(r, Math.min(g, b));
        float c = max - min;
        float h;
        if (c == 0) {
            h = 0;
        } else if (max == r) {
            h = (g - b) / c / 6f + (g >= b ? 0f : 1f);
        } else if (max == g) {
            h = ((b - r) / c + 2f) / 6f;
        } else {
            h = ((r - g) / c + 4f) / 6f;
        }
        return new float[]{h, max == 0 ? 0 : c / max, max};
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void largeImagesConvertEveryPixelAsAlone(Nd4jBackend backend) {
        for (long pixels : new long[]{5000, 100003}) {
            Nd4j.getRandom().setSeed(pixels);
            INDArray rgb = Nd4j.rand(DataType.FLOAT, pixels, 3);
            for (String op : new String[]{"rgb_to_hsv", "hsv_to_rgb"}) {
                INDArray whole = convert(op, rgb);
                // pieces of 500 pixels convert on one thread
                for (long start = 0; start < pixels; start += 500) {
                    long end = Math.min(pixels, start + 500);
                    INDArray piece = convert(op, rgb.get(NDArrayIndex.interval(start, end), NDArrayIndex.all()).dup());
                    assertEquals(piece, whole.get(NDArrayIndex.interval(start, end), NDArrayIndex.all()),
                            op + ", " + pixels + " pixels, rows " + start + " to " + end);
                }
            }
            INDArray hsv = convert("rgb_to_hsv", rgb);
            for (long p : new long[]{0, 1, 2047, 2048, pixels / 2, pixels - 1}) {
                float[] expected = rgbToHsv(rgb.getFloat(p, 0), rgb.getFloat(p, 1), rgb.getFloat(p, 2));
                for (int c = 0; c < 3; c++)
                    assertEquals(expected[c], hsv.getFloat(p, c), 1e-6, "pixel " + p + " channel " + c);
            }
        }
    }
}
