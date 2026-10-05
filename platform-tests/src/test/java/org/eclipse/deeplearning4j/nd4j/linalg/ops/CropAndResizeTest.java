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
import org.nd4j.linalg.api.ops.impl.image.CropAndResize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.tensorflow.framework.AttrValue;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * crop_and_resize against TensorFlow's definition: box (y1, x1, y2, x2) in [0, 1] units of the image it names, sampled
 * on a crop grid, bilinearly or at the nearest pixel, with the extrapolation value outside the image. The CPU helper
 * computed the sample positions in float for DOUBLE images (and the horizontal weight of an integer image as an
 * integer, which made it 0), left the crop of a box whose index named no image unwritten and read before the batch
 * for a negative index; the Java constructor without an output array handed the op a null one.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class CropAndResizeTest extends BaseNd4jTestWithBackends {

    private static final double[][] BOXES = {
            {0.0, 0.0, 1.0, 1.0},
            {0.1, 0.2, 0.9, 0.8},
            {0.5, 0.5, 0.5, 0.5},
            {-0.2, 0.3, 0.6, 1.2},
            {0.8, 0.1, 0.2, 0.9},
            {0.3, 0.3, 0.7, 0.7},
    };
    private static final int[] INDICES = {0, 1, 2, 2, 1, 0};

    private static INDArray boxes(DataType type) {
        return Nd4j.createFromArray(BOXES).castTo(type);
    }

    private static INDArray indices() {
        return Nd4j.createFromArray(INDICES);
    }

    private static INDArray image(DataType type) {
        Nd4j.getRandom().setSeed(77);
        INDArray random = Nd4j.rand(DataType.DOUBLE, 3, 8, 9, 2);
        return (type == DataType.FLOAT || type == DataType.DOUBLE) ? random.castTo(type)
                : random.muli(255).castTo(type);
    }

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    /** TensorFlow's crop_and_resize, in double precision. */
    private static double[] reference(INDArray image, int[] indices, int cropH, int cropW, boolean nearest,
                                      double extrapolation, boolean truncate) {
        int imageH = (int) image.size(1);
        int imageW = (int) image.size(2);
        int depth = (int) image.size(3);
        double[] out = new double[BOXES.length * cropH * cropW * depth];
        for (int b = 0; b < BOXES.length; b++) {
            double y1 = BOXES[b][0], x1 = BOXES[b][1], y2 = BOXES[b][2], x2 = BOXES[b][3];
            int in = indices[b];
            boolean inBatch = in >= 0 && in < image.size(0);
            double heightScale = cropH > 1 ? (y2 - y1) * (imageH - 1) / (cropH - 1) : 0;
            double widthScale = cropW > 1 ? (x2 - x1) * (imageW - 1) / (cropW - 1) : 0;
            for (int y = 0; y < cropH; y++) {
                double inY = cropH > 1 ? y1 * (imageH - 1) + y * heightScale : 0.5 * (y1 + y2) * (imageH - 1);
                for (int x = 0; x < cropW; x++) {
                    double inX = cropW > 1 ? x1 * (imageW - 1) + x * widthScale : 0.5 * (x1 + x2) * (imageW - 1);
                    for (int d = 0; d < depth; d++) {
                        double value;
                        if (!inBatch || inY < 0 || inY > imageH - 1 || inX < 0 || inX > imageW - 1) {
                            value = extrapolation;
                        } else if (nearest) {
                            value = image.getDouble(in, (long) Math.round(inY), (long) Math.round(inX), d);
                        } else {
                            int top = (int) Math.floor(inY), bottom = (int) Math.ceil(inY);
                            int left = (int) Math.floor(inX), right = (int) Math.ceil(inX);
                            double yLerp = inY - top, xLerp = inX - left;
                            double topValue = image.getDouble(in, top, left, d)
                                    + (image.getDouble(in, top, right, d) - image.getDouble(in, top, left, d)) * xLerp;
                            double bottomValue = image.getDouble(in, bottom, left, d)
                                    + (image.getDouble(in, bottom, right, d) - image.getDouble(in, bottom, left, d))
                                    * xLerp;
                            value = topValue + (bottomValue - topValue) * yLerp;
                        }
                        out[((b * cropH + y) * cropW + x) * depth + d] = truncate ? (double) (long) value : value;
                    }
                }
            }
        }
        return out;
    }

    private static INDArray crop(INDArray image, INDArray boxes, INDArray indices, int cropH, int cropW,
                                 boolean nearest, double extrapolation) {
        INDArray size = Nd4j.createFromArray(cropH, cropW);
        return Nd4j.exec(new CropAndResize(image, boxes, indices, size,
                nearest ? CropAndResize.Method.NEAREST : CropAndResize.Method.BILINEAR, extrapolation, null))[0];
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void cropRoundsTheProductBeforeAddingInItsComputationType(Nd4jBackend backend) {
        float fraction = 8191f / 8192f;
        assertEquals(-0x1.0p-16, Math.fma(1024.125f, fraction, -1024f), 0.0);
        double doubleFraction = 1.0 - 0x1.0p-27;
        assertEquals(-0x1.0p-28, Math.fma(0x1.0p26 + 0.5, doubleFraction, -0x1.0p26), 0.0);
        for (DataType imageType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (DataType boxType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                boolean doubleMath = imageType == DataType.DOUBLE || boxType == DataType.DOUBLE;
                // The FLOAT boxes case cannot encode doubleFraction; use FLOAT's exactly representable discriminator.
                double position = doubleMath && boxType == DataType.DOUBLE ? doubleFraction : fraction;
                double left = doubleMath && boxType == DataType.DOUBLE ? -0x1.0p26 : -1024.0;
                double right = doubleMath && boxType == DataType.DOUBLE ? 0.5 : 0.125;
                INDArray image = Nd4j.createFromArray(left, right).castTo(imageType).reshape(1, 1, 2, 1);
                INDArray boxes = Nd4j.createFromArray(new double[][]{{0.0, position, 0.0, position}}).castTo(boxType);
                INDArray output = crop(image, boxes, Nd4j.createFromArray(0), 1, 1, false, 7.0);
                assertEquals(imageType, output.dataType());
                // FLOAT image + FLOAT boxes uses stepwise FLOAT; DOUBLE image + FLOAT boxes retains DOUBLE precision.
                double expected = doubleMath && boxType == DataType.FLOAT ? -0x1.0p-16 : 0.0;
                assertEquals(expected, output.getDouble(0, 0, 0, 0), 0.0, imageType + "/" + boxType);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void nonfiniteCropPositionsGiveTheExtrapolationValue(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray image = Nd4j.ones(type, 1, 2, 2, 1);
            INDArray boxes = Nd4j.createFromArray(new double[][]{
                    {Double.NaN, 0.0, Double.NaN, 1.0},
                    {0.0, Double.NaN, 1.0, Double.NaN},
                    {Double.POSITIVE_INFINITY, 0.0, Double.POSITIVE_INFINITY, 1.0},
                    {0.0, Double.NEGATIVE_INFINITY, 1.0, Double.NEGATIVE_INFINITY}}).castTo(type);
            for (boolean nearest : new boolean[]{false, true}) {
                for (int[] size : new int[][]{{1, 1}, {2, 3}}) {
                    INDArray output = crop(image, boxes, Nd4j.createFromArray(0, 0, 0, 0), size[0], size[1], nearest, -7.0);
                    double[] expected = new double[(int) output.length()];
                    Arrays.fill(expected, -7.0);
                    assertArrayEquals(expected, logical(output), 0.0);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void outputDatatypeInferencePreservesTheImageDatatype(Nd4jBackend backend) {
        CropAndResize op = new CropAndResize();
        for (DataType imageType : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT32}) {
            assertEquals(Collections.singletonList(imageType), op.calculateOutputDataTypes(
                    Arrays.asList(imageType, DataType.DOUBLE, DataType.INT32, DataType.INT32)));
        }
        assertThrows(IllegalStateException.class, () -> op.calculateOutputDataTypes(
                Arrays.asList(DataType.FLOAT, DataType.INT32, DataType.INT32, DataType.INT32)));
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void tensorFlowImportRejectsUnknownInterpolationMethods(Nd4jBackend backend) {
        CropAndResize op = new CropAndResize();
        // An empty method is neither bilinear nor nearest and must not silently select bilinear.
        AttrValue method = AttrValue.getDefaultInstance();
        assertThrows(IllegalArgumentException.class, () -> op.initFromTensorFlow(null, null,
                Collections.singletonMap("method", method), null));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void doubleCropsAreAccurateToDoublePrecision(Nd4jBackend backend) {
        INDArray image = image(DataType.DOUBLE);
        // nearest rounds a position, so it is checked on the grids of these boxes that have no position on a half
        // pixel, where a rounding of the last bit could change the pixel
        int[][] bilinearGrids = {{6, 5}, {1, 1}, {1, 4}, {7, 1}, {13, 11}};
        int[][] nearestGrids = {{6, 5}, {1, 1}, {1, 4}};
        for (boolean nearest : new boolean[]{false, true}) {
            for (int[] size : nearest ? nearestGrids : bilinearGrids) {
                for (double extrapolation : new double[]{0.0, -1.5}) {
                    INDArray out = crop(image, boxes(DataType.DOUBLE), indices(), size[0], size[1], nearest,
                            extrapolation);
                    assertArrayEquals(new long[]{BOXES.length, size[0], size[1], 2}, out.shape());
                    double[] expected = reference(image, INDICES, size[0], size[1], nearest, extrapolation, false);
                    assertArrayEquals(expected, logical(out), 1e-12, "DOUBLE crop " + size[0] + " x " + size[1]
                            + (nearest ? ", nearest" : ", bilinear") + ", extrapolation " + extrapolation);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void floatBilinearCropsMatchTheDefinition(Nd4jBackend backend) {
        INDArray image = image(DataType.FLOAT);
        INDArray out = crop(image, boxes(DataType.FLOAT), indices(), 6, 5, false, 0.25);
        assertArrayEquals(reference(image, INDICES, 6, 5, false, 0.25, false), logical(out), 1e-5);
    }

    /** The weights of an integer image's bilinear crop are fractions: its crop interpolates, then truncates. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void integerImagesAreInterpolated(Nd4jBackend backend) {
        INDArray image = image(DataType.INT32);
        INDArray out = crop(image, boxes(DataType.FLOAT), indices(), 6, 5, false, 0.0);
        // a value that lands on an integer may round either way: within one is the interpolation, the old horizontal
        // weight of 0 ignored the right column
        assertArrayEquals(reference(image, INDICES, 6, 5, false, 0.0, true), logical(out), 1.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyLayoutGivesTheCOrderCrops(Nd4jBackend backend) {
        INDArray image = image(DataType.DOUBLE);
        INDArray expected = crop(image.dup('c'), boxes(DataType.DOUBLE).dup('c'), indices(), 6, 5, false, 0.0);
        INDArray parent = Nd4j.zeros(DataType.DOUBLE, 6, 16, 18, 4).addi(-7.0);
        INDArray stepped = parent.get(NDArrayIndex.interval(0, 2, 6), NDArrayIndex.interval(0, 2, 16),
                NDArrayIndex.interval(0, 2, 18), NDArrayIndex.interval(0, 2, 4));
        stepped.assign(image);
        for (INDArray variant : new INDArray[]{image.dup('f'), stepped}) {
            INDArray out = crop(variant, boxes(DataType.DOUBLE).dup('f'), indices(), 6, 5, false, 0.0);
            assertArrayEquals(logical(expected), logical(out), 1e-13);
        }
    }

    /**
     * A box naming an image outside the batch, or before it, is outside every image: its crop is the extrapolation
     * value. Its crop used to be left unwritten, and an index below zero read before the batch.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void boxIndicesOutsideTheBatchGiveTheExtrapolationValue(Nd4jBackend backend) {
        INDArray image = image(DataType.DOUBLE);
        int[] indices = {0, 3, 1, -1, 2, 7};
        for (boolean nearest : new boolean[]{false, true}) {
            INDArray out = crop(image, boxes(DataType.DOUBLE), Nd4j.createFromArray(indices), 4, 3, nearest, -2.5);
            assertArrayEquals(reference(image, indices, 4, 3, nearest, -2.5, false), logical(out), 1e-12,
                    nearest ? "nearest" : "bilinear");
        }
    }
}
