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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * rgb_to_hsv and hsv_to_rgb convert every pixel of a large image as they convert it alone. The CPU fast path stepped
 * through the elements by 3 in chunks Threads::parallel_for split by element count, not on the pixel grid: from about
 * 2048 pixels, threads after the first mixed channels of neighbouring pixels and the last read and wrote past the end.
 * hsv_to_rgb held the hue's sector in a float whatever the element type, so a DOUBLE conversion was only as accurate
 * as float. Every layout converts as its C-order copy does.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class ColorSpaceConversionTest extends BaseNd4jTestWithBackends {

    private static final DataType[] FLOATING = {DataType.FLOAT, DataType.DOUBLE};

    private static INDArray convert(String op, INDArray in) {
        return Nd4j.exec(DynamicCustomOp.builder(op).addInputs(in).build())[0];
    }

    private static INDArray convert(String op, INDArray in, long channelAxis) {
        return Nd4j.exec(DynamicCustomOp.builder(op).addInputs(in).addIntegerArguments(channelAxis).build())[0];
    }

    /** TensorFlow's RGB to HSV (adjust_hue.h), hue in [0, 1). */
    private static double[] rgbToHsv(double r, double g, double b) {
        double max = Math.max(r, Math.max(g, b));
        double min = Math.min(r, Math.min(g, b));
        double c = max - min;
        double h;
        if (c == 0) {
            h = 0;
        } else if (max == r) {
            h = (g - b) / c / 6.0 + (g >= b ? 0.0 : 1.0);
        } else if (max == g) {
            h = ((b - r) / c + 2.0) / 6.0;
        } else {
            h = ((r - g) / c + 4.0) / 6.0;
        }
        return new double[]{h, max == 0 ? 0 : c / max, max};
    }

    /** TensorFlow's HSV to RGB (adjust_hue.h) for a hue in [0, 1). */
    private static double[] hsvToRgb(double h, double s, double v) {
        double sector = h * 6.0;
        double c = v * s;
        if (sector < 1) {
            return new double[]{v, v - c * (1 - sector), v - c};
        } else if (sector < 2) {
            return new double[]{v - c * (sector - 1), v, v - c};
        } else if (sector < 3) {
            return new double[]{v - c, v, v - c * (3 - sector)};
        } else if (sector < 4) {
            return new double[]{v - c, v - c * (sector - 3), v};
        } else if (sector < 5) {
            return new double[]{v - c * (5 - sector), v - c, v};
        }
        return new double[]{v, v - c, v - c * (sector - 5)};
    }

    private static double tolerance(DataType type) {
        return type == DataType.DOUBLE ? 1e-12 : 1e-6;
    }

    /** The logical (C order) content of an array, whatever its layout. */
    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void largeImagesConvertEveryPixelAsAlone(Nd4jBackend backend) {
        for (DataType type : FLOATING) {
            for (long pixels : new long[]{5000, 100003}) {
                Nd4j.getRandom().setSeed(pixels);
                INDArray rgb = Nd4j.rand(type, pixels, 3);
                for (String op : new String[]{"rgb_to_hsv", "hsv_to_rgb"}) {
                    INDArray whole = convert(op, rgb);
                    // pieces of 500 pixels convert on one thread
                    for (long start = 0; start < pixels; start += 500) {
                        long end = Math.min(pixels, start + 500);
                        INDArray piece = convert(op, rgb.get(NDArrayIndex.interval(start, end), NDArrayIndex.all()).dup());
                        assertEquals(piece, whole.get(NDArrayIndex.interval(start, end), NDArrayIndex.all()),
                                type + " " + op + ", " + pixels + " pixels, rows " + start + " to " + end);
                    }
                }
                INDArray hsv = convert("rgb_to_hsv", rgb);
                for (long p : new long[]{0, 1, 2047, 2048, pixels / 2, pixels - 1}) {
                    double[] expected = rgbToHsv(rgb.getDouble(p, 0), rgb.getDouble(p, 1), rgb.getDouble(p, 2));
                    for (int c = 0; c < 3; c++)
                        assertEquals(expected[c], hsv.getDouble(p, c), tolerance(type),
                                type + " pixel " + p + " channel " + c);
                }
            }
        }
    }

    /**
     * A DOUBLE conversion is accurate to double precision: the sector of the hue, h * 6, was a float, which put errors
     * of about 1e-7 into every output channel.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void doubleConversionsKeepDoublePrecision(Nd4jBackend backend) {
        long pixels = 4099;
        Nd4j.getRandom().setSeed(7);
        INDArray hsv = Nd4j.rand(DataType.DOUBLE, pixels, 3);
        INDArray rgb = convert("hsv_to_rgb", hsv);
        for (long p = 0; p < pixels; p++) {
            double[] expected = hsvToRgb(hsv.getDouble(p, 0), hsv.getDouble(p, 1), hsv.getDouble(p, 2));
            for (int c = 0; c < 3; c++)
                assertEquals(expected[c], rgb.getDouble(p, c), 1e-13, "hsv_to_rgb pixel " + p + " channel " + c);
        }

        // and a trip there and back returns what it started from
        Nd4j.getRandom().setSeed(8);
        INDArray original = Nd4j.rand(DataType.DOUBLE, pixels, 3);
        INDArray back = convert("hsv_to_rgb", convert("rgb_to_hsv", original));
        assertArrayEquals(logical(original), logical(back), 1e-13, "rgb -> hsv -> rgb");
    }

    /** Every layout of the input converts as the C-order copy of it does, along the channel axis it names. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyLayoutConvertsAsItsContiguousCopy(Nd4jBackend backend) {
        for (DataType type : FLOATING) {
            Nd4j.getRandom().setSeed(11);
            INDArray pixels = Nd4j.rand(type, 5000, 3);
            INDArray image = Nd4j.rand(type, 2, 25, 41, 3);
            INDArray planar = Nd4j.rand(type, 3, 4001);
            INDArray parent = Nd4j.rand(type, 10001, 3);
            INDArray stepped = parent.get(NDArrayIndex.interval(1, 2, 10001), NDArrayIndex.all());

            for (String op : new String[]{"rgb_to_hsv", "hsv_to_rgb"}) {
                String where = type + " " + op;
                // F order, and the same pixels in a stepped view of a larger array
                assertArrayEquals(logical(convert(op, pixels.dup('c'))), logical(convert(op, pixels.dup('f'))),
                        tolerance(type), where + ", [5000, 3] in F order");
                assertArrayEquals(logical(convert(op, stepped.dup('c'))), logical(convert(op, stepped)),
                        tolerance(type), where + ", [5000, 3] stepped view");
                assertArrayEquals(logical(convert(op, image.dup('c'))), logical(convert(op, image.dup('f'))),
                        tolerance(type), where + ", [2, 25, 41, 3] in F order");
                // channels first, and channels named by a negative axis
                assertArrayEquals(logical(convert(op, planar.dup('c'), 0)), logical(convert(op, planar.dup('f'), 0)),
                        tolerance(type), where + ", [3, 4001], channel axis 0, F order");
                assertArrayEquals(logical(convert(op, planar.permute(1, 0).dup('c'), 1)),
                        logical(convert(op, planar.dup('c'), 0).permute(1, 0)),
                        tolerance(type), where + ", the same pixels with the channel axis last and first");
                assertArrayEquals(logical(convert(op, image.dup('c'), -1)), logical(convert(op, image.dup('c'), 3)),
                        tolerance(type), where + ", channel axis -1");
            }
        }
    }

    /** The conversions run in place: the pixels' own channels are read before they are overwritten. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void inPlaceConversionsEqualOutOfPlaceOnes(Nd4jBackend backend) {
        for (DataType type : FLOATING) {
            Nd4j.getRandom().setSeed(13);
            INDArray pixels = Nd4j.rand(type, 4097, 3);
            for (String op : new String[]{"rgb_to_hsv", "hsv_to_rgb"}) {
                INDArray expected = convert(op, pixels);
                INDArray inPlace = pixels.dup();
                Nd4j.exec(DynamicCustomOp.builder(op).addInputs(inPlace).addOutputs(inPlace).build());
                assertArrayEquals(logical(expected), logical(inPlace), tolerance(type), type + " " + op + " in place");
            }
        }
    }
}
