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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Random;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The image resizes (nearest neighbor, bilinear, bicubic, area) and crop_and_resize work on [batch, height, width,
 * channels] images of any layout (C or F order, offset and stepped views, permuted views) and give the results of the
 * algorithms of TensorFlow, which the tests below compute by plain loops.
 *
 * The CUDA nearest neighbor kernel launched a block per image with a thread per output pixel (swapped): images past
 * the number of output pixels were never written and a batch of 1025 or more was an invalid launch; its images were
 * read from the buffer start, ignoring a view's offset. The bilinear, bicubic and area kernels assumed dense C-order
 * images, the bicubic one needed one thread per block, overflowed its shared memory from 33 channels, raced across
 * blocks from 129 output columns and handled 100 channels as 3; the area one sized its row cache by the output width;
 * resize_images with the bicubic method wrote nothing at all, on either backend.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class ImageResizeParityTest extends BaseNd4jTestWithBackends {

    // ------------------------------------------------------------------------------------------------ layouts

    /** What a parent array holds around a view: a value no operation may read. */
    private static double filler(DataType type) {
        return type.isFPType() ? Double.NaN : -7777;
    }

    private static final class Layout {
        final String name;
        final UnaryOperator<INDArray> of;

        Layout(String name, UnaryOperator<INDArray> of) {
            this.name = name;
            this.of = of;
        }
    }

    /** A view of the same elements that is stepped along an axis (starting past the buffer's start). */
    private static INDArray steppedAlong(INDArray x, int axis) {
        long[] s = x.shape().clone();
        long n = s[axis];
        s[axis] = 2 * n + 1;
        INDArray parent = Nd4j.valueArrayOf(s, filler(x.dataType()), x.dataType());
        INDArrayIndex[] idx = new INDArrayIndex[s.length];
        for (int i = 0; i < s.length; i++)
            idx[i] = i == axis ? NDArrayIndex.interval(1, 2, 2 * n + 1) : NDArrayIndex.all();
        return parent.get(idx).assign(x);
    }

    private static final Layout[] LAYOUTS = {
            new Layout("C order", x -> x.dup('c')),
            new Layout("F order", x -> x.dup('f')),
            new Layout("offset view", x -> {
                long[] s = x.shape().clone();
                s[0] += 1;
                INDArray parent = Nd4j.valueArrayOf(s, filler(x.dataType()), x.dataType());
                INDArrayIndex[] idx = new INDArrayIndex[s.length];
                idx[0] = NDArrayIndex.interval(1, s[0]);
                for (int i = 1; i < s.length; i++)
                    idx[i] = NDArrayIndex.all();
                return parent.get(idx).assign(x);
            }),
            new Layout("view stepped along the rows", x -> steppedAlong(x, 1)),
            new Layout("view stepped along the columns", x -> steppedAlong(x, 2)),
            new Layout("view stepped along the channels", x -> steppedAlong(x, 3)),
            new Layout("permuted view", x -> {
                long[] s = x.shape();
                int r = s.length;
                long[] reversed = new long[r];
                long[] perm = new long[r];
                for (int i = 0; i < r; i++) {
                    reversed[i] = s[r - 1 - i];
                    perm[i] = r - 1 - i;
                }
                return Nd4j.create(x.dataType(), reversed, 'c').permute(perm).assign(x);
            }),
    };

    /** An array of the shape, as a view of a larger parent of NaN (or a filler), shifted past the buffer's start. */
    private static INDArray viewOf(DataType type, long... shape) {
        long[] s = shape.clone();
        s[0] += 1;
        s[s.length - 1] = 2 * shape[shape.length - 1] + 1;
        INDArray parent = Nd4j.valueArrayOf(s, filler(type), type);
        INDArrayIndex[] idx = new INDArrayIndex[s.length];
        idx[0] = NDArrayIndex.interval(1, s[0]);
        for (int i = 1; i < s.length - 1; i++)
            idx[i] = NDArrayIndex.all();
        idx[s.length - 1] = NDArrayIndex.interval(1, 2, s[s.length - 1]);
        return parent.get(idx);
    }

    // ------------------------------------------------------------------------------------------------ data

    private static long size(long... shape) {
        long n = 1;
        for (long s : shape)
            n *= s;
        return n;
    }

    /** Multiples of 1/8 below 128 in size: exact in every type the ops handle. */
    private static INDArray data(Random random, DataType type, long... shape) {
        double[] values = new double[(int) size(shape)];
        for (int i = 0; i < values.length; i++)
            values[i] = Math.floor(random.nextDouble() * 2000 - 1000) / 8;
        return Nd4j.createFromArray(values).reshape(shape).castTo(type);
    }

    /** The elements of an array in C order, whatever its layout. */
    private static double[] flat(INDArray a) {
        return a.dup('c').castTo(DataType.DOUBLE).data().asDouble();
    }

    private static INDArray sizeArray(int height, int width) {
        return Nd4j.createFromArray(new int[]{height, width});
    }

    private static void assertClose(double[] expected, INDArray actual, double tolerance, String what) {
        double[] a = flat(actual);
        assertEquals(expected.length, a.length, what + ": number of elements");
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(expected[i] - a[i]) > tolerance * (1 + Math.abs(expected[i])))
                throw new AssertionError(what + ": element " + i + " is " + a[i] + ", expected " + expected[i]);
        }
    }

    // ------------------------------------------------------------------------------------------------ references

    private interface Scaler {
        float apply(int x, float scale);
    }

    private static final Scaler LEGACY = (x, scale) -> (float) x * scale;
    private static final Scaler HALF_PIXEL = (x, scale) -> ((float) x + 0.5f) * scale - 0.5f;
    private static final Scaler HALF_PIXEL_NN = (x, scale) -> ((float) x + 0.5f) * scale;

    private static float resizeScale(long in, long out, boolean alignCorners) {
        return (alignCorners && out > 1) ? (in - 1) / (float) (out - 1) : in / (float) out;
    }

    /** C's roundf: halves away from zero. */
    private static double roundf(double v) {
        return v >= 0 ? Math.floor(v + 0.5) : -Math.floor(-v + 0.5);
    }

    /** nearest modes: 0 floor, 1 round prefer floor, 2 round prefer ceil, 3 ceil */
    private static int nearestSource(Scaler scaler, int out, float scale, int mode, long inSize, boolean halfPixel) {
        float v = scaler.apply(out, scale);
        double r;
        switch (mode) {
            case 1:
                r = v == (float) ((long) v) + 0.5f ? Math.floor(v) : roundf(v);
                break;
            case 2:
                r = roundf(v);
                break;
            case 3:
                r = Math.ceil(v);
                break;
            default:
                r = Math.floor(v);
        }
        long index = Math.min((long) r, inSize - 1);
        if (halfPixel)
            index = Math.max(0, index);
        return (int) index;
    }

    private static double[] nearestReference(double[] in, int b, int h, int w, int c, int oh, int ow, Scaler scaler,
                                             int mode, boolean alignCorners, boolean halfPixel) {
        float hs = resizeScale(h, oh, alignCorners);
        float ws = resizeScale(w, ow, alignCorners);
        double[] out = new double[b * oh * ow * c];
        for (int y = 0; y < oh; y++) {
            int iy = nearestSource(scaler, y, hs, mode, h, halfPixel);
            for (int x = 0; x < ow; x++) {
                int ix = nearestSource(scaler, x, ws, mode, w, halfPixel);
                for (int bb = 0; bb < b; bb++)
                    for (int ch = 0; ch < c; ch++)
                        out[((bb * oh + y) * ow + x) * c + ch] = in[((bb * h + iy) * w + ix) * c + ch];
            }
        }
        return out;
    }

    private static double[] bilinearReference(double[] in, int b, int h, int w, int c, int oh, int ow,
                                              boolean alignCorners, boolean halfPixel) {
        float hs = resizeScale(h, oh, alignCorners);
        float ws = resizeScale(w, ow, alignCorners);
        Scaler scaler = halfPixel ? HALF_PIXEL : LEGACY;
        // lower index, upper index and interpolation value of every output row and column
        double[][] ys = new double[oh][3];
        double[][] xs = new double[ow][3];
        for (int i = 0; i < oh; i++)
            interpolation(scaler, i, hs, h, ys[i]);
        for (int i = 0; i < ow; i++)
            interpolation(scaler, i, ws, w, xs[i]);
        double[] out = new double[b * oh * ow * c];
        for (int bb = 0; bb < b; bb++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int ch = 0; ch < c; ch++) {
                        double tl = in[((bb * h + (int) ys[y][0]) * w + (int) xs[x][0]) * c + ch];
                        double tr = in[((bb * h + (int) ys[y][0]) * w + (int) xs[x][1]) * c + ch];
                        double bl = in[((bb * h + (int) ys[y][1]) * w + (int) xs[x][0]) * c + ch];
                        double br = in[((bb * h + (int) ys[y][1]) * w + (int) xs[x][1]) * c + ch];
                        double top = tl + (tr - tl) * xs[x][2];
                        double bottom = bl + (br - bl) * xs[x][2];
                        out[((bb * oh + y) * ow + x) * c + ch] = top + (bottom - top) * ys[y][2];
                    }
        return out;
    }

    private static void interpolation(Scaler scaler, int i, float scale, long inSize, double[] into) {
        double in = scaler.apply(i, scale);
        double floor = Math.floor(in);
        double ceil = Math.ceil(in);
        into[0] = Math.max((long) floor, 0);
        into[1] = Math.min((long) ceil, inSize - 1);
        into[2] = in - floor;
    }

    private static final int TABLE_SIZE = 1 << 10;

    /** The cubic convolution table of the bicubic resize, in float like the native code. */
    private static float[] coefficientTable(float a) {
        float[] table = new float[(TABLE_SIZE + 1) * 2];
        for (int i = 0; i <= TABLE_SIZE; i++) {
            float x = (float) (i * 1.0 / TABLE_SIZE);
            table[i * 2] = ((a + 2f) * x - (a + 3f)) * x * x + 1f;
            x = (float) (x + 1.0);
            table[i * 2 + 1] = ((a * x - 5f * a) * x + 8f * a) * x - 4f * a;
        }
        return table;
    }

    private static int bound(long v, long limit) {
        return (int) Math.min(limit - 1, Math.max(0, v));
    }

    /** The four source indices and weights of an output row or column of the bicubic resize. */
    private static void bicubicWeights(float[] table, Scaler scaler, int outLoc, float scale, long limit,
                                       boolean excludeOutside, int[] index, float[] weight) {
        float inLocF = scaler.apply(outLoc, scale);
        long inLoc = (long) Math.floor(inLocF);
        float delta = inLocF - inLoc;
        int offset = (int) roundf(delta * TABLE_SIZE);
        if (excludeOutside) {
            index[0] = bound(inLoc - 1, limit);
            weight[0] = index[0] == inLoc - 1 ? table[offset * 2 + 1] : 0f;
            index[1] = bound(inLoc, limit);
            weight[1] = index[1] == inLoc ? table[offset * 2] : 0f;
            index[2] = bound(inLoc + 1, limit);
            weight[2] = index[2] == inLoc + 1 ? table[(TABLE_SIZE - offset) * 2] : 0f;
            index[3] = bound(inLoc + 2, limit);
            weight[3] = index[3] == inLoc + 2 ? table[(TABLE_SIZE - offset) * 2 + 1] : 0f;
            float sum = weight[0] + weight[1] + weight[2] + weight[3];
            if (Math.abs(sum) >= 1000f * Float.MIN_NORMAL) {
                float inverse = 1f / sum;
                for (int i = 0; i < 4; i++)
                    weight[i] *= inverse;
            }
        } else {
            weight[0] = table[offset * 2 + 1];
            weight[1] = table[offset * 2];
            weight[2] = table[(TABLE_SIZE - offset) * 2];
            weight[3] = table[(TABLE_SIZE - offset) * 2 + 1];
            index[0] = bound(inLoc - 1, limit);
            index[1] = bound(inLoc, limit);
            index[2] = bound(inLoc + 1, limit);
            index[3] = bound(inLoc + 2, limit);
        }
    }

    private static float[] bicubicReference(double[] in, int b, int h, int w, int c, int oh, int ow, boolean alignCorners,
                                            boolean halfPixel, boolean excludeOutside, float coefficient) {
        float hs = resizeScale(h, oh, alignCorners);
        float ws = resizeScale(w, ow, alignCorners);
        Scaler scaler = halfPixel ? HALF_PIXEL : LEGACY;
        float[] table = coefficientTable(coefficient);
        int[][] yIndex = new int[oh][4];
        float[][] yWeight = new float[oh][4];
        int[][] xIndex = new int[ow][4];
        float[][] xWeight = new float[ow][4];
        for (int i = 0; i < oh; i++)
            bicubicWeights(table, scaler, i, hs, h, excludeOutside, yIndex[i], yWeight[i]);
        for (int i = 0; i < ow; i++)
            bicubicWeights(table, scaler, i, ws, w, excludeOutside, xIndex[i], xWeight[i]);
        float[] out = new float[b * oh * ow * c];
        for (int bb = 0; bb < b; bb++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int ch = 0; ch < c; ch++) {
                        float[] column = new float[4];
                        for (int i = 0; i < 4; i++) {
                            float v = 0f;
                            for (int j = 0; j < 4; j++)
                                v += (float) in[((bb * h + yIndex[y][j]) * w + xIndex[x][i]) * c + ch] * yWeight[y][j];
                            column[i] = v;
                        }
                        float r = 0f;
                        for (int i = 0; i < 4; i++)
                            r += column[i] * xWeight[x][i];
                        out[((bb * oh + y) * ow + x) * c + ch] = r;
                    }
        return out;
    }

    private static double[] toDouble(float[] values) {
        double[] d = new double[values.length];
        for (int i = 0; i < d.length; i++)
            d[i] = values[i];
        return d;
    }

    private static double[] areaReference(double[] in, int b, int h, int w, int c, int oh, int ow, boolean alignCorners) {
        float hs = resizeScale(h, oh, alignCorners);
        float ws = resizeScale(w, ow, alignCorners);
        double scale = 1.0 / ((double) hs * (double) ws);
        double[] out = new double[b * oh * ow * c];
        for (int y = 0; y < oh; y++) {
            float inY = y * hs;
            float inY1 = (y + 1) * hs;
            long yStart = (long) Math.floor(inY);
            long yEnd = (long) Math.ceil(inY1);
            for (int x = 0; x < ow; x++) {
                float inX = x * ws;
                float inX1 = (x + 1) * ws;
                long xStart = (long) Math.floor(inX);
                long xEnd = (long) Math.ceil(inX1);
                double startScale = areaEdgeScale(xStart, inX, inX1, ws);
                double endScale = areaEdgeScale(xEnd - 1, inX, inX1, ws);
                for (int bb = 0; bb < b; bb++)
                    for (int ch = 0; ch < c; ch++) {
                        double total = 0;
                        for (long i = yStart; i < yEnd; i++) {
                            double scaleY = i < inY ? (i + 1 > inY1 ? hs : i + 1 - inY) : (i + 1 > inY1 ? inY1 - i : 1.0);
                            int row = bound(i, h);
                            double sumY = in[((bb * h + row) * w + bound(xStart, w)) * c + ch] * startScale;
                            if (xStart + 1 != xEnd) {
                                for (long xx = xStart + 1; xx < xEnd - 1; xx++)
                                    sumY += in[((bb * h + row) * w + bound(xx, w)) * c + ch];
                                sumY += in[((bb * h + row) * w + bound(xEnd - 1, w)) * c + ch] * endScale;
                            }
                            total += sumY * scaleY;
                        }
                        out[((bb * oh + y) * ow + x) * c + ch] = total * scale;
                    }
            }
        }
        return out;
    }

    /** The weight of the input column v in the output column [inX, inX1) of the area resize. */
    private static double areaEdgeScale(long v, float inX, float inX1, float widthScale) {
        return v < inX ? (v + 1 > inX1 ? widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.0);
    }

    // ------------------------------------------------------------------------------------------------ nearest

    private static INDArray nearest(INDArray images, int height, int width, boolean alignCorners, boolean halfPixel,
                                    INDArray out) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("resize_nearest_neighbor")
                .addInputs(images, sizeArray(height, width)).addBooleanArguments(alignCorners, halfPixel);
        if (out != null)
            builder.addOutputs(out);
        return Nd4j.exec(builder.build())[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nearestMatchesTheGoldensOfTheNativeTests(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 24).reshape(1, 2, 3, 4);
        double[] floor = {
                1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12,
                1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24,
                13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24};
        double[] halfPixel = {
                1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12,
                1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12,
                13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24,
                13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24};

        // the size as integer arguments, height first
        INDArray byArguments = Nd4j.exec(DynamicCustomOp.builder("resize_nearest_neighbor").addInputs(input)
                .addIntegerArguments(4, 5).addBooleanArguments(false, false).build())[0];
        assertArrayEquals(new long[]{1, 4, 5, 4}, byArguments.shape());
        assertArrayEquals(floor, flat(byArguments), 0.0, "size by arguments");

        INDArray bySize = nearest(input, 4, 5, false, false, null);
        assertArrayEquals(floor, flat(bySize), 0.0, "size by array");

        INDArray centers = nearest(input, 4, 5, false, true, null);
        assertArrayEquals(halfPixel, flat(centers), 0.0, "half pixel centers");

        // a 3D image is a batch of one
        INDArray image = input.reshape(2, 3, 4);
        INDArray resized = Nd4j.exec(DynamicCustomOp.builder("resize_nearest_neighbor").addInputs(image)
                .addIntegerArguments(4, 5).addBooleanArguments(false, false).build())[0];
        assertArrayEquals(new long[]{4, 5, 4}, resized.shape());
        assertArrayEquals(floor, flat(resized), 0.0, "3D image");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nearestWritesEveryImageOfALargeBatch(Nd4jBackend backend) {
        Random random = new Random(21);
        // outputs of one to four pixels against batches up to 2049: the kernel launched a block per output pixel with a
        // thread per image, so images past the pixel count were never written and 1025 images were an invalid launch
        for (int batch : new int[]{1, 2, 5, 17, 300, 1025, 2049}) {
            for (int[] out : new int[][]{{1, 1}, {2, 2}, {1, 4}, {4, 2}}) {
                INDArray images = data(random, DataType.FLOAT, batch, 4, 4, 3);
                double[] values = flat(images);
                // asymmetric (floor), half pixel centers, align corners (round prefer ceil)
                boolean[][] modes = {{false, false}, {false, true}, {true, false}};
                for (boolean[] mode : modes) {
                    Scaler scaler = mode[1] ? HALF_PIXEL_NN : LEGACY;
                    double[] expected = nearestReference(values, batch, 4, 4, 3, out[0], out[1], scaler,
                            mode[0] ? 2 : 0, mode[0], mode[1]);
                    INDArray result = nearest(images, out[0], out[1], mode[0], mode[1], null);
                    assertArrayEquals(new long[]{batch, out[0], out[1], 3}, result.shape());
                    assertArrayEquals(expected, flat(result), 0.0, batch + " images to " + Arrays.toString(out)
                            + ", align corners " + mode[0] + ", half pixel centers " + mode[1]);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nearestKeepsItsTypes(Nd4jBackend backend) {
        Random random = new Random(22);
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16, DataType.INT32, DataType.INT64}) {
            INDArray images = data(random, type, 3, 8, 4, 2);
            double[] expected = nearestReference(flat(images), 3, 8, 4, 2, 4, 8, LEGACY, 0, false, false);
            INDArray result = nearest(images, 4, 8, false, false, null);
            assertEquals(type, result.dataType());
            assertArrayEquals(expected, flat(result), 0.0, type.toString());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nearestOfEveryCoordinateModeAndNearestMode(Nd4jBackend backend) {
        Random random = new Random(23);
        // coordinate modes 0 asymmetric, 1 half pixel, 2 half pixel (nearest); nearest modes 0 floor, 1 round prefer
        // floor, 2 round prefer ceil, 3 ceil. The scales (2, 0.5, 1.5, 0.25) are exact in float.
        int[][] sizes = {{8, 8, 4, 4}, {4, 4, 8, 8}, {6, 6, 4, 4}, {4, 8, 16, 2}};
        for (int[] s : sizes) {
            INDArray images = data(random, DataType.FLOAT, 5, s[0], s[1], 2);
            double[] values = flat(images);
            for (int coordinate = 0; coordinate < 3; coordinate++) {
                for (int mode = 0; mode < 4; mode++) {
                    Scaler scaler = coordinate == 0 ? LEGACY : coordinate == 1 ? HALF_PIXEL : HALF_PIXEL_NN;
                    double[] expected = nearestReference(values, 5, s[0], s[1], 2, s[2], s[3], scaler, mode, false,
                            coordinate != 0);
                    INDArray result = Nd4j.exec(DynamicCustomOp.builder("image_resize").addInputs(images, sizeArray(s[2], s[3]))
                            .addIntegerArguments(1, coordinate, mode).addBooleanArguments(false, false).build())[0];
                    assertArrayEquals(new long[]{5, s[2], s[3], 2}, result.shape());
                    assertArrayEquals(expected, flat(result), 0.0, Arrays.toString(s) + ", coordinate mode "
                            + coordinate + ", nearest mode " + mode);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nearestReadsAndWritesViews(Nd4jBackend backend) {
        Random random = new Random(24);
        INDArray images = data(random, DataType.DOUBLE, 3, 8, 4, 2);
        double[] values = flat(images);
        double[] expected = nearestReference(values, 3, 8, 4, 2, 4, 8, LEGACY, 0, false, false);
        for (Layout layout : LAYOUTS) {
            INDArray input = layout.of.apply(images);
            assertArrayEquals(expected, flat(nearest(input, 4, 8, false, false, null)), 0.0, "input in " + layout.name);
            // the same, written into a view of a larger array: the rest of the parent stays as it was
            INDArray target = viewOf(DataType.DOUBLE, 3, 4, 8, 2);
            nearest(input, 4, 8, false, false, target);
            assertArrayEquals(expected, flat(target), 0.0, "output view, input in " + layout.name);
        }
        // the first image of an offset view, resized alone
        INDArray big = data(random, DataType.DOUBLE, 5, 8, 4, 2);
        INDArray slice = big.get(NDArrayIndex.interval(2, 3), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        assertArrayEquals(nearestReference(flat(slice), 1, 8, 4, 2, 4, 8, LEGACY, 0, false, false),
                flat(nearest(slice, 4, 8, false, false, null)), 0.0, "one image of a batch");
    }

    // ------------------------------------------------------------------------------------------------ bilinear

    private static INDArray bilinear(INDArray images, int height, int width, boolean alignCorners, boolean halfPixel,
                                     INDArray out) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("resize_bilinear")
                .addInputs(images, sizeArray(height, width)).addBooleanArguments(alignCorners, halfPixel);
        if (out != null)
            builder.addOutputs(out);
        return Nd4j.exec(builder.build())[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bilinearMatchesTheReference(Nd4jBackend backend) {
        Random random = new Random(31);
        int[][] cases = {{1, 4, 5, 3, 9, 7}, {3, 7, 6, 2, 4, 3}, {2, 3, 3, 1, 11, 13}, {4, 9, 9, 4, 9, 9}, {1025, 3, 3, 1, 2, 2},
                {2, 6, 300, 2, 5, 40}, {1, 5, 4, 100, 7, 6}};
        boolean[][] modes = {{false, false}, {true, false}, {false, true}};
        for (int[] c : cases) {
            INDArray images = data(random, DataType.FLOAT, c[0], c[1], c[2], c[3]);
            double[] values = flat(images);
            for (boolean[] mode : modes) {
                double[] expected = bilinearReference(values, c[0], c[1], c[2], c[3], c[4], c[5], mode[0], mode[1]);
                INDArray result = bilinear(images, c[4], c[5], mode[0], mode[1], null);
                assertEquals(DataType.FLOAT, result.dataType());
                assertClose(expected, result, 1e-5, Arrays.toString(c) + ", align corners " + mode[0]
                        + ", half pixel centers " + mode[1]);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bilinearReadsAndWritesViews(Nd4jBackend backend) {
        Random random = new Random(32);
        INDArray images = data(random, DataType.FLOAT, 3, 7, 6, 3);
        double[] expected = bilinearReference(flat(images), 3, 7, 6, 3, 9, 4, false, false);
        for (Layout layout : LAYOUTS) {
            INDArray input = layout.of.apply(images);
            assertClose(expected, bilinear(input, 9, 4, false, false, null), 1e-5, "input in " + layout.name);
            INDArray target = viewOf(DataType.FLOAT, 3, 9, 4, 3);
            bilinear(input, 9, 4, false, false, target);
            assertClose(expected, target, 1e-5, "output view, input in " + layout.name);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bilinearOfOtherTypes(Nd4jBackend backend) {
        Random random = new Random(33);
        // doubles stay doubles, integers give floats
        for (DataType type : new DataType[]{DataType.DOUBLE, DataType.INT32}) {
            INDArray images = data(random, type, 2, 5, 5, 2);
            double[] expected = bilinearReference(flat(images), 2, 5, 5, 2, 8, 7, false, true);
            INDArray result = bilinear(images, 8, 7, false, true, null);
            assertEquals(type == DataType.DOUBLE ? DataType.DOUBLE : DataType.FLOAT, result.dataType());
            assertClose(expected, result, 1e-5, type.toString());
        }
    }

    // ------------------------------------------------------------------------------------------------ bicubic

    private static INDArray bicubic(INDArray images, int height, int width, boolean alignCorners, boolean halfPixel,
                                    INDArray out) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("resize_bicubic")
                .addInputs(images, sizeArray(height, width)).addBooleanArguments(alignCorners, halfPixel);
        if (out != null)
            builder.addOutputs(out);
        return Nd4j.exec(builder.build())[0];
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void bicubicPreservesSingletonImagesAtBothSupportedRanks(Nd4jBackend backend) {
        for (int rank : new int[]{3, 4}) {
            long[] inputShape = rank == 3 ? new long[]{1, 1, 1} : new long[]{1, 1, 1, 1};
            for (DataType inputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                INDArray image = Nd4j.valueArrayOf(inputShape, 2.5, inputType);
                for (int mode = 0; mode < 3; mode++) {
                    for (int[] size : new int[][]{{1, 3}, {3, 1}, {1, 1}}) {
                        long[] outputShape = rank == 3 ? new long[]{size[0], size[1], 1}
                                : new long[]{1, size[0], size[1], 1};
                        INDArray allocated = bicubic(image, size[0], size[1], mode == 1, mode == 2, null);
                        assertArrayEquals(outputShape, allocated.shape());
                        assertEquals(DataType.FLOAT, allocated.dataType());
                        assertEquals(Nd4j.valueArrayOf(outputShape, 2.5, DataType.FLOAT), allocated);
                        for (DataType outputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                            INDArray output = Nd4j.valueArrayOf(outputShape, Double.NaN, outputType);
                            bicubic(image, size[0], size[1], mode == 1, mode == 2, output);
                            assertEquals(Nd4j.valueArrayOf(outputShape, 2.5, outputType), output);
                        }
                    }
                }
            }
        }
    }

    /** Independent unit impulses expose each table coefficient without interpolation cancellation. */
    @ParameterizedTest
    @MethodSource("configs")
    public void bicubicRoundsCoefficientPolynomialsBeforeInterpolation(Nd4jBackend backend) {
        // With align corners, 9 -> 15 has scale 4/7. Row 8 samples rows [3,4,5,6]
        // at table offset 585, the same nonexact coefficients as the retained random regression.
        float[] table = coefficientTable(-0.75f);
        int[] entries = {1171, 1170, 878, 879};
        for (DataType inputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray image = Nd4j.zeros(inputType, 1, 9, 1, 4);
            for (int channel = 0; channel < 4; channel++)
                image.putScalar(new long[]{0, 3 + channel, 0, channel}, 1.0);
            INDArray allocated = bicubic(image, 15, 1, true, false, null);
            assertEquals(DataType.FLOAT, allocated.dataType());
            for (int channel = 0; channel < 4; channel++)
                assertEquals((double) table[entries[channel]], allocated.getDouble(0, 8, 0, channel), 0.0);
            for (DataType outputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                INDArray output = Nd4j.valueArrayOf(new long[]{1, 15, 1, 4}, Double.NaN, outputType);
                bicubic(image, 15, 1, true, false, output);
                assertEquals(outputType, output.dataType());
                for (int channel = 0; channel < 4; channel++)
                    assertEquals((double) table[entries[channel]], output.getDouble(0, 8, 0, channel), 0.0);
            }
        }
    }

    /** At x=3 the source position is exactly 1.5 and the Keys weights are [-3,19,19,-3]/32. */
    @ParameterizedTest
    @MethodSource("configs")
    public void bicubicRoundsEachProductBeforeAdding(Nd4jBackend backend) {
        float value = 1f + 0x1.0p-23f;
        float negativeProduct = -value * (19f / 32f);
        // Contracting the positive product with the rounded negative one leaves a nonzero residual.
        assertEquals(3.0 * 0x1.0p-28, Math.fma(value, 19f / 32f, negativeProduct), 0.0);
        for (DataType inputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray image = Nd4j.createFromArray(0f, -value, value, 0f).castTo(inputType).reshape(1, 1, 4, 1);
            INDArray allocated = bicubic(image, 1, 7, true, false, null);
            assertEquals(DataType.FLOAT, allocated.dataType());
            assertEquals(0.0, allocated.getDouble(0, 0, 3, 0), 0.0);
            for (DataType outputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                INDArray output = Nd4j.valueArrayOf(new long[]{1, 1, 7, 1}, Double.NaN, outputType);
                bicubic(image, 1, 7, true, false, output);
                assertEquals(outputType, output.dataType());
                assertEquals(0.0, output.getDouble(0, 0, 3, 0), 0.0);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bicubicMatchesTheGoldenOfTheNativeTests(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 36).reshape(1, 3, 3, 4);
        double[] expected = {
                1.000000, 2.000000, 3.000000, 4.000000, 2.625000, 3.625000, 4.625000, 5.625000, 5.000000, 6.000000, 7.000000, 8.000000,
                7.375000, 8.375000, 9.375000, 10.375000, 9.000000, 10.000000, 11.000000, 12.000000, 9.375000, 10.375000, 11.375000, 12.375000,
                5.875000, 6.875000, 7.875000, 8.875000, 7.500000, 8.500000, 9.500000, 10.500000, 9.875000, 10.875000, 11.875000, 12.875000,
                12.250000, 13.250000, 14.250000, 15.250000, 13.875000, 14.875000, 15.875000, 16.875000, 14.250000, 15.250000, 16.250000, 17.250000,
                13.000000, 14.000000, 15.000000, 16.000000, 14.625000, 15.625000, 16.625000, 17.625000, 17.000000, 18.000000, 19.000000, 20.000000,
                19.375000, 20.375000, 21.375000, 22.375000, 21.000000, 22.000000, 23.000000, 24.000000, 21.375000, 22.375000, 23.375000, 24.375000,
                20.125000, 21.125000, 22.125000, 23.125000, 21.750000, 22.750000, 23.750000, 24.750000, 24.125000, 25.125000, 26.125000, 27.125000,
                26.500000, 27.500000, 28.500000, 29.500000, 28.125000, 29.125000, 30.125000, 31.125000, 28.500000, 29.500000, 30.500000, 31.500000,
                25.000000, 26.000000, 27.000000, 28.000000, 26.625000, 27.625000, 28.625000, 29.625000, 29.000000, 30.000000, 31.000000, 32.000000,
                31.375000, 32.375000, 33.375000, 34.375000, 33.000000, 34.000000, 35.000000, 36.000000, 33.375000, 34.375000, 35.375000, 36.375000,
                26.125000, 27.125000, 28.125000, 29.125000, 27.750000, 28.750000, 29.750000, 30.750000, 30.125000, 31.125000, 32.125000, 33.125000,
                32.500000, 33.500000, 34.500000, 35.500000, 34.125000, 35.125000, 36.125000, 37.125000, 34.500000, 35.500000, 36.500000, 37.500000};
        INDArray result = bicubic(input, 6, 6, false, false, null);
        assertArrayEquals(new long[]{1, 6, 6, 4}, result.shape());
        assertClose(expected, result, 1e-5, "3 x 3 x 4 to 6 x 6");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bicubicMatchesTheReference(Nd4jBackend backend) {
        Random random = new Random(41);
        // channels 3 (unrolled on the CPU), 1, 4, 40 (past the 33 the CUDA kernel's shared memory held) and 100 (the
        // kernel handled it as 3); 300 output columns (past the 128 of its single serial block)
        int[][] cases = {{2, 5, 6, 3, 7, 9}, {1, 6, 5, 1, 4, 4}, {3, 4, 4, 4, 9, 6}, {1, 5, 6, 40, 8, 5}, {1, 4, 5, 100, 6, 7},
                {1, 6, 20, 2, 8, 300}, {1025, 4, 4, 1, 5, 5}};
        // align corners and half pixel centers: 0 neither (the legacy mode: coefficient -0.75), 1 align corners, 2 half
        // pixel centers (coefficient -0.5, the borders excluded)
        for (int[] c : cases) {
            INDArray images = data(random, DataType.FLOAT, c[0], c[1], c[2], c[3]);
            double[] values = flat(images);
            for (int mode = 0; mode < 3; mode++) {
                boolean align = mode == 1;
                boolean halfPixel = mode == 2;
                float[] expected = bicubicReference(values, c[0], c[1], c[2], c[3], c[4], c[5], align, halfPixel,
                        halfPixel, halfPixel ? -0.5f : -0.75f);
                INDArray result = bicubic(images, c[4], c[5], align, halfPixel, null);
                assertEquals(DataType.FLOAT, result.dataType());
                assertClose(toDouble(expected), result, 5e-5, Arrays.toString(c) + ", mode " + mode);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bicubicReadsImagesOfAnyLayout(Nd4jBackend backend) {
        Random random = new Random(42);
        INDArray images = data(random, DataType.FLOAT, 3, 5, 6, 3);
        float[] expected = bicubicReference(flat(images), 3, 5, 6, 3, 9, 8, false, false, false, -0.75f);
        for (Layout layout : LAYOUTS)
            assertClose(toDouble(expected), bicubic(layout.of.apply(images), 9, 8, false, false, null), 5e-5,
                    "input in " + layout.name);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bicubicWritesADoubleOutput(Nd4jBackend backend) {
        // the op allows FLOAT32 and DOUBLE outputs; a DOUBLE one was written as floats (the interpolation is in float on
        // both backends, so the DOUBLE output holds the FLOAT32 output's values)
        Random random = new Random(45);
        INDArray images = data(random, DataType.FLOAT, 2, 5, 6, 3);
        for (int mode = 0; mode < 3; mode++) {
            boolean align = mode == 1;
            boolean halfPixel = mode == 2;
            INDArray asFloats = bicubic(images, 8, 9, align, halfPixel, null);
            assertEquals(DataType.FLOAT, asFloats.dataType());
            INDArray doubles = Nd4j.valueArrayOf(new long[]{2, 8, 9, 3}, Double.NaN, DataType.DOUBLE);
            bicubic(images, 8, 9, align, halfPixel, doubles);
            assertEquals(DataType.DOUBLE, doubles.dataType());
            assertClose(flat(asFloats), doubles, 1e-6, "mode " + mode);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void imageResizeBicubicOfEveryCoordinateMode(Nd4jBackend backend) {
        Random random = new Random(43);
        INDArray images = data(random, DataType.FLOAT, 2, 5, 6, 3);
        double[] values = flat(images);
        // the unified image_resize op: method 2 is bicubic; coordinate modes 0 asymmetric, 1 half pixel, 2 half pixel
        // (nearest). The arguments are the bicubic coefficient, and booleans (unused, antialias, exclude outside).
        for (int coordinate = 0; coordinate < 2; coordinate++) {
            for (boolean excludeOutside : new boolean[]{false, true}) {
                float coefficient = coordinate == 1 ? -0.5f : -0.75f;
                float[] expected = bicubicReference(values, 2, 5, 6, 3, 8, 9, false, coordinate == 1, excludeOutside,
                        coefficient);
                INDArray result = Nd4j.exec(DynamicCustomOp.builder("image_resize").addInputs(images, sizeArray(8, 9))
                        .addIntegerArguments(2, coordinate).addBooleanArguments(false, false, excludeOutside)
                        .addFloatingPointArguments((double) coefficient).build())[0];
                assertClose(toDouble(expected), result, 5e-5, "coordinate mode " + coordinate + ", exclude outside "
                        + excludeOutside);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void resizeImagesWithTheBicubicMethodWritesItsOutput(Nd4jBackend backend) {
        // resize_images with method 2 (bicubic) wrote nothing on either backend
        Random random = new Random(44);
        INDArray images = data(random, DataType.FLOAT, 2, 5, 6, 3);
        double[] values = flat(images);
        for (boolean alignCorners : new boolean[]{false, true}) {
            float[] expected = bicubicReference(values, 2, 5, 6, 3, 8, 9, alignCorners, false, false, -0.75f);
            INDArray out = Nd4j.valueArrayOf(new long[]{2, 8, 9, 3}, Double.NaN, DataType.FLOAT);
            Nd4j.exec(DynamicCustomOp.builder("resize_images").addInputs(images, sizeArray(8, 9)).addIntegerArguments(2)
                    .addBooleanArguments(alignCorners).addOutputs(out).build());
            assertClose(toDouble(expected), out, 5e-5, "align corners " + alignCorners);
        }
    }

    // ------------------------------------------------------------------------------------------------ area

    private static INDArray area(INDArray images, int height, int width, boolean alignCorners) {
        return Nd4j.exec(DynamicCustomOp.builder("resize_area").addInputs(images, sizeArray(height, width))
                .addBooleanArguments(alignCorners).build())[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void areaMatchesTheGoldensOfTheNativeTests(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 9).reshape(1, 3, 3, 1);
        double[] doubled = {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 7, 7, 8, 8, 9, 9};
        assertClose(doubled, area(input, 6, 6, false), 1e-5, "3 x 3 to 6 x 6");
        double[] aligned = {1, 1, 1.5, 2, 2, 3, 1, 1, 1.5, 2, 2, 3, 2.5, 2.5, 3, 3.5, 3.5, 4.5, 4, 4, 4.5, 5, 5, 6, 4, 4, 4.5, 5, 5, 6,
                7, 7, 7.5, 8, 8, 9};
        assertClose(aligned, area(input, 6, 6, true), 1e-5, "3 x 3 to 6 x 6 with align corners");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void areaMatchesTheReference(Nd4jBackend backend) {
        Random random = new Random(51);
        // the last two shrink 64 rows to 2 and 1 against 1 output column: an output row reads 32 input rows, so the CUDA
        // kernel's row cache (sized by the output width) overflowed into the next row's and then past the array's end
        int[][] cases = {{2, 7, 5, 3, 4, 3}, {1, 6, 6, 1, 12, 9}, {3, 9, 10, 4, 3, 5}, {2, 64, 8, 3, 2, 1}, {1, 64, 8, 2, 1, 1},
                {1025, 4, 4, 1, 2, 2}, {1, 10, 300, 2, 4, 7}};
        for (int[] c : cases) {
            for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT32}) {
                INDArray images = data(random, type, c[0], c[1], c[2], c[3]);
                double[] values = flat(images);
                for (boolean align : new boolean[]{false, true}) {
                    double[] expected = areaReference(values, c[0], c[1], c[2], c[3], c[4], c[5], align);
                    INDArray result = area(images, c[4], c[5], align);
                    assertEquals(DataType.FLOAT, result.dataType());
                    assertClose(expected, result, 1e-4, type + " " + Arrays.toString(c) + ", align corners " + align);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void areaReadsImagesOfAnyLayout(Nd4jBackend backend) {
        Random random = new Random(52);
        INDArray images = data(random, DataType.FLOAT, 3, 8, 6, 3);
        double[] expected = areaReference(flat(images), 3, 8, 6, 3, 4, 5, false);
        // an F-order input gives an F-order output, which the kernel could not write as the dense array it assumed
        for (Layout layout : LAYOUTS)
            assertClose(expected, area(layout.of.apply(images), 4, 5, false), 1e-4, "input in " + layout.name);
    }

    // ------------------------------------------------------------------------------------------------ crop and resize

    private static float[] cropAndResizeReference(double[] images, int batch, int h, int w, int c, float[][] boxes,
                                                  int[] indices, int cropH, int cropW, int method, float extrapolation) {
        float[] out = new float[boxes.length * cropH * cropW * c];
        for (int b = 0; b < boxes.length; b++) {
            float y1 = boxes[b][0], x1 = boxes[b][1], y2 = boxes[b][2], x2 = boxes[b][3];
            int bIn = indices[b];
            float heightScale = cropH > 1 ? (y2 - y1) * (h - 1) / (cropH - 1) : 0f;
            float widthScale = cropW > 1 ? (x2 - x1) * (w - 1) / (cropW - 1) : 0f;
            for (int y = 0; y < cropH; y++) {
                float inY = cropH > 1 ? y1 * (h - 1) + y * heightScale : (float) (0.5 * (y1 + y2) * (h - 1));
                for (int x = 0; x < cropW; x++) {
                    float inX = cropW > 1 ? x1 * (w - 1) + x * widthScale : (float) (0.5 * (x1 + x2) * (w - 1));
                    for (int d = 0; d < c; d++) {
                        int at = ((b * cropH + y) * cropW + x) * c + d;
                        if (inY < 0 || inY > h - 1 || inX < 0 || inX > w - 1) {
                            out[at] = extrapolation;
                        } else if (method == 0) {
                            int top = (int) Math.floor(inY);
                            int bottom = (int) Math.ceil(inY);
                            float yLerp = inY - top;
                            int left = (int) Math.floor(inX);
                            int right = (int) Math.ceil(inX);
                            float xLerp = inX - left;
                            float tl = (float) images[((bIn * h + top) * w + left) * c + d];
                            float tr = (float) images[((bIn * h + top) * w + right) * c + d];
                            float bl = (float) images[((bIn * h + bottom) * w + left) * c + d];
                            float br = (float) images[((bIn * h + bottom) * w + right) * c + d];
                            float t = tl + (tr - tl) * xLerp;
                            float bo = bl + (br - bl) * xLerp;
                            out[at] = t + (bo - t) * yLerp;
                        } else {
                            int closestX = (int) roundf(inX);
                            int closestY = (int) roundf(inY);
                            out[at] = (float) images[((bIn * h + closestY) * w + closestX) * c + d];
                        }
                    }
                }
            }
        }
        return out;
    }

    /** True when a sample position of the box's crop grid is within 1e-3 of a half pixel. */
    private static boolean nearAHalfPixel(float[] box, int h, int w, int cropH, int cropW) {
        return nearAHalf(box[0], box[2], h, cropH) || nearAHalf(box[1], box[3], w, cropW);
    }

    private static boolean nearAHalf(double lo, double hi, int size, int crop) {
        double scale = crop > 1 ? (hi - lo) * (size - 1) / (crop - 1) : 0;
        for (int i = 0; i < crop; i++) {
            double position = crop > 1 ? lo * (size - 1) + i * scale : 0.5 * (lo + hi) * (size - 1);
            if (Math.abs(position - Math.floor(position) - 0.5) < 1e-3)
                return true;
        }
        return false;
    }

    private static INDArray cropAndResize(INDArray images, INDArray boxes, INDArray indices, int cropH, int cropW,
                                          int method, double extrapolation) {
        return Nd4j.exec(DynamicCustomOp.builder("crop_and_resize").addInputs(images, boxes, indices, sizeArray(cropH, cropW))
                .addIntegerArguments(method).addFloatingPointArguments(extrapolation).build())[0];
    }

    private static void assertInvalidCrop(INDArray images, INDArray boxes, INDArray indices, INDArray size,
                                          long... methods) {
        DynamicCustomOp shapeOp = DynamicCustomOp.builder("crop_and_resize").addInputs(images, boxes, indices, size)
                .addIntegerArguments(methods).build();
        assertThrows(RuntimeException.class, shapeOp::calculateOutputShape, "shape inference must reject the contract");
        DynamicCustomOp executeOp = DynamicCustomOp.builder("crop_and_resize").addInputs(images, boxes, indices, size)
                .addIntegerArguments(methods).build();
        assertThrows(RuntimeException.class, () -> Nd4j.exec(executeOp), "execution must reject the contract");
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void cropAndResizeRejectsInvalidContractsBeforeShapeReads(Nd4jBackend backend) {
        INDArray image = Nd4j.ones(DataType.FLOAT, 1, 2, 2, 1);
        INDArray boxes = Nd4j.createFromArray(new float[][]{{0f, 0f, 1f, 1f}});
        INDArray indices = Nd4j.createFromArray(0);
        INDArray size = sizeArray(1, 1);
        assertInvalidCrop(image.reshape(2, 2), boxes, indices, size, 0);
        assertInvalidCrop(image, boxes.reshape(4), indices, size, 0);
        assertInvalidCrop(image, Nd4j.zeros(DataType.FLOAT, 1, 3), indices, size, 0);
        assertInvalidCrop(image, boxes.castTo(DataType.INT32), indices, size, 0);
        assertInvalidCrop(image, boxes, indices.castTo(DataType.FLOAT), size, 0);
        assertInvalidCrop(image, boxes, indices, size.castTo(DataType.FLOAT), 0);
        assertInvalidCrop(image, Nd4j.zeros(DataType.FLOAT, 2, 4), indices, size, 0);
        assertInvalidCrop(image, boxes, indices, sizeArray(0, 1), 0);
        assertInvalidCrop(image, boxes, indices, sizeArray(1, -1), 0);
        assertInvalidCrop(image, boxes, indices, Nd4j.createFromArray(1), 0);
        // Validate before narrowing: 2^32 used to become method 0 at the helper boundary.
        for (long method : new long[]{-1, 2, 1L << 32})
            assertInvalidCrop(image, boxes, indices, size, method);
        assertInvalidCrop(image, boxes, indices, size, 0, 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cropAndResizeMatchesTheReference(Nd4jBackend backend) {
        Random random = new Random(61);
        // (images, height, width, channels, boxes, crop height, crop width): more boxes than images, and 1030 images
        int[][] cases = {{3, 6, 5, 3, 7, 4, 5}, {1, 5, 5, 1, 2, 7, 7}, {3, 8, 6, 2, 1500, 3, 3}, {1030, 3, 3, 1, 4, 2, 2},
                {2, 9, 7, 4, 5, 1, 1}};
        for (int[] c : cases) {
            INDArray images = data(random, DataType.FLOAT, c[0], c[1], c[2], c[3]);
            float[][] boxes = new float[c[4]][4];
            int[] indices = new int[c[4]];
            for (int b = 0; b < c[4]; b++) {
                // inside the image; every fifth box reaches past it (the pixels outside are the extrapolation value)
                if (b % 5 == 4) {
                    boxes[b] = new float[]{-0.5f, -0.75f, 1.5f, 1.25f};
                } else {
                    // no sample position on a half pixel: the nearest method could round either way there, depending
                    // on how the backend rounds the position's last bit
                    do {
                        float y1 = 0.1f + random.nextFloat() * 0.3f;
                        float x1 = 0.1f + random.nextFloat() * 0.3f;
                        boxes[b] = new float[]{y1, x1, y1 + 0.2f + random.nextFloat() * 0.4f,
                                x1 + 0.2f + random.nextFloat() * 0.4f};
                    } while (nearAHalfPixel(boxes[b], c[1], c[2], c[5], c[6]));
                }
                indices[b] = random.nextInt(c[0]);
            }
            double[] values = flat(images);
            INDArray boxArray = Nd4j.createFromArray(boxes);
            INDArray indexArray = Nd4j.createFromArray(indices);
            for (int method = 0; method < 2; method++) {
                float[] expected = cropAndResizeReference(values, c[0], c[1], c[2], c[3], boxes, indices, c[5], c[6], method,
                        -3.5f);
                INDArray result = cropAndResize(images, boxArray, indexArray, c[5], c[6], method, -3.5);
                assertArrayEquals(new long[]{c[4], c[5], c[6], c[3]}, result.shape());
                assertClose(toDouble(expected), result, 1e-5, Arrays.toString(c) + ", method " + method);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cropAndResizeReadsImagesAndIndicesOfAnyLayout(Nd4jBackend backend) {
        Random random = new Random(62);
        INDArray images = data(random, DataType.FLOAT, 4, 7, 6, 3);
        float[][] boxes = {{0.1f, 0.1f, 0.8f, 0.9f}, {0.25f, 0.3f, 0.6f, 0.7f}, {-0.5f, -0.25f, 1.5f, 1.25f}};
        int[] indices = {3, 0, 2};
        double[] values = flat(images);
        float[] expected = cropAndResizeReference(values, 4, 7, 6, 3, boxes, indices, 5, 4, 0, 0f);
        INDArray indexView = steppedAlong(Nd4j.createFromArray(indices), 0);
        for (Layout layout : LAYOUTS)
            assertClose(toDouble(expected), cropAndResize(layout.of.apply(images), Nd4j.createFromArray(boxes), indexView,
                    5, 4, 0, 0.0), 1e-5, "images in " + layout.name);
        assertTrue(indexView.stride(0) != 1, "the indices are a stepped view");
    }
}
