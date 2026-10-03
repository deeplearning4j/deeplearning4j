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
import org.nd4j.linalg.api.ops.custom.DrawBoundingBoxes;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * draw_bounding_boxes draws each box's frame in a color from the table, cycling through its rows (colors), a later
 * box over an earlier one where they overlap, as TensorFlow does. The CUDA helper cycled through the table's length
 * (colors times channels), reading past its last color, drew a block's boxes in no particular order and read views
 * from their buffers' starts; the op skipped execution on an empty input (an empty color table, images without
 * boxes), leaving its output unwritten.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class DrawBoundingBoxesTest extends BaseNd4jTestWithBackends {

    // TensorFlow's default colors
    private static final float[][] DEFAULT_COLORS = {{1, 1, 0, 1}, {0, 0, 1, 1}, {1, 0, 0, 1}, {0, 1, 0, 1},
            {0.5f, 0, 0.5f, 1}, {0.5f, 0.5f, 0, 1}, {0.5f, 0, 0, 1}, {0, 0, 0.5f, 1}, {0, 1, 1, 1}, {1, 0, 1, 1}};

    /** TensorFlow's default colors for an image depth: white (first channel 1) for gray images. */
    private static float[][] defaultColors(long depth) {
        float[][] colors = new float[DEFAULT_COLORS.length][];
        for (int i = 0; i < colors.length; i++) {
            colors[i] = DEFAULT_COLORS[i].clone();
            if (depth == 1)
                colors[i][0] = 1;
        }
        return colors;
    }

    /** TensorFlow's DrawBoundingBoxes: each box's frame, in box order, clipped to the image. */
    private static INDArray reference(INDArray images, INDArray boxes, float[][] colors) {
        INDArray out = images.castTo(DataType.DOUBLE).dup('c');
        long batch = images.size(0), height = images.size(1), width = images.size(2), depth = images.size(3);
        for (long b = 0; b < batch; b++) {
            for (long k = 0; k < boxes.size(1); k++) {
                float[] color = colors[(int) (k % colors.length)];
                long rowStart = (long) ((height - 1) * boxes.getFloat(b, k, 0));
                long colStart = (long) ((width - 1) * boxes.getFloat(b, k, 1));
                long rowEnd = (long) ((height - 1) * boxes.getFloat(b, k, 2));
                long colEnd = (long) ((width - 1) * boxes.getFloat(b, k, 3));
                if (rowStart > rowEnd || colStart > colEnd)
                    continue;
                if (rowStart >= height || rowEnd < 0 || colStart >= width || colEnd < 0)
                    continue;
                long rowStartBound = Math.max(0, rowStart), rowEndBound = Math.min(height - 1, rowEnd);
                long colStartBound = Math.max(0, colStart), colEndBound = Math.min(width - 1, colEnd);
                for (long j = colStartBound; j <= colEndBound; j++) {
                    if (rowStart >= 0)
                        put(out, b, rowStart, j, color, depth);
                    if (rowEnd < height)
                        put(out, b, rowEnd, j, color, depth);
                }
                for (long i = rowStartBound; i <= rowEndBound; i++) {
                    if (colStart >= 0)
                        put(out, b, i, colStart, color, depth);
                    if (colEnd < width)
                        put(out, b, i, colEnd, color, depth);
                }
            }
        }
        return out.castTo(images.dataType());
    }

    private static void put(INDArray out, long b, long i, long j, float[] color, long depth) {
        for (long c = 0; c < depth; c++)
            out.putScalar(new long[]{b, i, j, c}, color[(int) c]);
    }

    private static INDArray draw(INDArray images, INDArray boxes, INDArray colors) {
        INDArray out = Nd4j.create(images.dataType(), images.shape());
        Nd4j.exec(new DrawBoundingBoxes(images, boxes, colors, out));
        return out;
    }

    /** Images whose values differ from each other and from the colors. */
    private static INDArray images(DataType dataType, long... shape) {
        long length = 1;
        for (long s : shape)
            length *= s;
        return Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, length).divi(4 * length).reshape(shape).castTo(dataType);
    }

    // a frame over the image's border, inner frames, an inverted box, a box partly and a box completely outside
    private static final float[][] BOXES = {{0, 0, 1, 1}, {0.2f, 0.1f, 0.8f, 0.7f}, {0.6f, 0.5f, 0.3f, 0.9f},
            {-0.2f, -0.3f, 0.5f, 1.4f}, {1.2f, 0, 1.5f, 1}, {0.35f, 0.25f, 0.95f, 0.55f}};

    private static INDArray boxes(long batch) {
        float[][][] boxes = new float[(int) batch][][];
        for (int b = 0; b < batch; b++) {
            boxes[b] = new float[BOXES.length][];
            for (int k = 0; k < BOXES.length; k++) {
                // each image its own boxes: shift the inner ones
                boxes[b][k] = BOXES[k].clone();
                if (k == 1 || k == 5)
                    for (int c = 0; c < 4; c++)
                        boxes[b][k][c] += 0.05f * b;
            }
        }
        return Nd4j.createFromArray(boxes);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void moreBoxesThanColorsCycleThroughTheColors(Nd4jBackend backend) {
        for (long depth : new long[]{1, 3, 4}) {
            INDArray images = images(DataType.FLOAT, 2, 9, 11, depth);
            INDArray boxes = boxes(2);
            // two colors for six boxes, with more channels than the images have
            float[][] colors = {{2.25f, 2.5f, 2.75f, 2.875f}, {3.25f, 3.5f, 3.75f, 3.875f}};
            INDArray out = draw(images, boxes, Nd4j.createFromArray(colors));
            assertEquals(reference(images, boxes, colors), out, "depth " + depth);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void laterBoxesDrawOverEarlierOnes(Nd4jBackend backend) {
        // wide frames starting at different columns: a pixel's colors come from threads of different warps
        INDArray images = images(DataType.FLOAT, 2, 70, 300, 4);
        float[][] boxRows = {{0, 0, 1, 1}, {0, 0.1f, 0.9f, 0.95f}, {0.05f, 0.02f, 1, 0.7f}, {0, 0.33f, 0.5f, 1},
                {0.2f, 0, 0.8f, 0.5f}, {0, 0.07f, 1, 0.93f}, {0, 0.21f, 0.7f, 0.66f}};
        INDArray boxes = Nd4j.createFromArray(new float[][][]{boxRows, boxRows});
        float[][] colors = new float[boxRows.length][];
        for (int k = 0; k < colors.length; k++)
            colors[k] = new float[]{2 + k, 3 + k, 4 + k, 5 + k};
        INDArray expected = reference(images, boxes, colors);
        INDArray colorsArr = Nd4j.createFromArray(colors);
        for (int run = 0; run < 10; run++)
            assertEquals(expected, draw(images, boxes, colorsArr), "run " + run);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void anEmptyColorTableDrawsTheDefaultColors(Nd4jBackend backend) {
        for (long depth : new long[]{1, 3, 4}) {
            INDArray images = images(DataType.FLOAT, 1, 12, 13, depth);
            // twelve boxes: the default table's ten colors and two more
            float[][][] boxRows = new float[1][12][];
            for (int k = 0; k < 12; k++)
                boxRows[0][k] = new float[]{0.05f * k, 0.04f * k, 0.5f + 0.04f * k, 0.45f + 0.05f * k};
            INDArray boxes = Nd4j.createFromArray(boxRows);
            INDArray out = draw(images, boxes, Nd4j.create(DataType.FLOAT, 0, 4));
            assertEquals(reference(images, boxes, defaultColors(depth)), out, "depth " + depth);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void imagesWithoutBoxesAreCopied(Nd4jBackend backend) {
        // the op did not run on an empty input: the output stayed unwritten
        INDArray colors = Nd4j.createFromArray(new float[][]{{2.25f, 2.5f, 2.75f}});
        INDArray images = images(DataType.FLOAT, 2, 9, 11, 3);
        assertEquals(images, draw(images, Nd4j.create(DataType.FLOAT, 2, 0, 4), colors), "no boxes");
        INDArray noImages = Nd4j.create(DataType.FLOAT, 0, 9, 11, 3);
        INDArray out = draw(noImages, Nd4j.create(DataType.FLOAT, 0, 2, 4), colors);
        assertTrue(out.isEmpty(), "no images");
        assertArrayEquals(noImages.shape(), out.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void halfImages(Nd4jBackend backend) {
        INDArray images = images(DataType.HALF, 2, 9, 11, 3);
        INDArray boxes = boxes(2);
        float[][] colors = {{0.25f, 0.5f, 0.75f}, {0.125f, 0.375f, 0.625f}, {0.875f, 0.0625f, 0.9375f}};
        INDArray out = draw(images, boxes, Nd4j.createFromArray(colors));
        assertEquals(DataType.HALF, out.dataType());
        assertEquals(reference(images, boxes, colors), out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void viewsAreReadFromTheirOffsets(Nd4jBackend backend) {
        float[][] colorRows = {{9, 9, 9}, {2.25f, 2.5f, 2.75f}, {3.25f, 3.5f, 3.75f}, {9, 9, 9}};
        // views past their buffers' starts
        INDArray images = images(DataType.FLOAT, 4, 9, 11, 3).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray boxes = boxes(3).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray colors = Nd4j.createFromArray(colorRows).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        float[][] used = {colorRows[1], colorRows[2]};
        assertEquals(reference(images.dup(), boxes.dup(), used), draw(images, boxes, colors), "offset views");

        // strided views: three of four channels, four of six box values, three of four color channels
        INDArray stridedImages = images(DataType.FLOAT, 2, 9, 11, 4).get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.interval(1, 4));
        INDArray wideBoxes = Nd4j.zeros(DataType.FLOAT, 2, BOXES.length, 6);
        wideBoxes.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 5)).assign(boxes(2));
        INDArray stridedBoxes = wideBoxes.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 5));
        INDArray wideColors = Nd4j.createFromArray(new float[][]{{9, 2.25f, 2.5f, 2.75f}, {9, 3.25f, 3.5f, 3.75f}});
        INDArray stridedColors = wideColors.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 4));
        assertEquals(reference(stridedImages.dup(), stridedBoxes.dup(), stridedColors.dup().toFloatMatrix()),
                draw(stridedImages, stridedBoxes, stridedColors), "strided views");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void rejectsADepthOrBoxWidthTensorFlowDoesNotDraw(Nd4jBackend backend) {
        INDArray colors = Nd4j.createFromArray(new float[][]{{0.5f, 0.25f, 0.75f, 1}});
        INDArray twoChannels = images(DataType.FLOAT, 1, 5, 5, 2);
        INDArray boxes = boxes(1);
        RuntimeException depth = assertThrows(RuntimeException.class, () -> draw(twoChannels, boxes, colors));
        assertTrue(String.valueOf(depth.getMessage()).contains("draw_bounding_boxes"), "depth 2: " + depth.getMessage());
        INDArray threeChannels = images(DataType.FLOAT, 1, 5, 5, 3);
        INDArray narrowBoxes = Nd4j.createFromArray(new float[][][]{{{0.1f, 0.1f, 0.9f}}});
        RuntimeException width = assertThrows(RuntimeException.class,
                () -> draw(threeChannels, narrowBoxes, colors));
        assertTrue(String.valueOf(width.getMessage()).contains("draw_bounding_boxes"),
                "three box coordinates: " + width.getMessage());
    }
}
