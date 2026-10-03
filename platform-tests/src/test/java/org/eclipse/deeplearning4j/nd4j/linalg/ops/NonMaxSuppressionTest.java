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
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * non_max_suppression returns the indices of the boxes it selects, in decreasing score order, and leaves the scores
 * as they were. Its output held as many indices as the maximum allowed (or the boxes scoring above the threshold),
 * leaving the tail unwritten when overlap suppression selected fewer; the CUDA helper clamped and sorted the scores
 * input in place.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class NonMaxSuppressionTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void returnsTheBoxesItSelects(Nd4jBackend backend) {
        // TensorFlow's three clusters: overlap suppression keeps three boxes whatever the larger maximum
        INDArray boxes = Nd4j.createFromArray(new float[][]{{0, 0, 1, 1}, {0, 0.1f, 1, 1.1f}, {0, -0.1f, 1, 0.9f},
                {0, 10, 1, 11}, {0, 10.1f, 1, 11.1f}, {0, 100, 1, 101}});
        INDArray scores = Nd4j.createFromArray(new float[]{0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
        INDArray scoresBefore = scores.dup();
        INDArray expected = Nd4j.createFromArray(new int[]{3, 0, 5});
        for (int maxOutputSize : new int[]{3, 6}) {
            INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("non_max_suppression").addInputs(boxes, scores)
                    .addIntegerArguments(maxOutputSize).addFloatingPointArguments(0.5).build());
            assertEquals(expected, out[0].castTo(DataType.INT32), "non_max_suppression, at most " + maxOutputSize);
            assertTrue(scoresBefore.equalsWithEps(scores, 0.0), "the scores changed: " + scores);
        }
    }

    /** Greedy selection: boxes by decreasing score, each kept unless its IoU with a kept box exceeds the threshold. */
    private static int[] greedyReference(float[][] boxes, float[] scores, int maxOutputSize, double iouThreshold) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));
        List<Integer> kept = new ArrayList<>();
        for (int idx : order) {
            if (kept.size() >= maxOutputSize)
                break;
            boolean suppressed = false;
            for (int k : kept) {
                if (iou(boxes[idx], boxes[k]) > iouThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed)
                kept.add(idx);
        }
        return kept.stream().mapToInt(Integer::intValue).toArray();
    }

    private static double iou(float[] a, float[] b) {
        double aMinY = Math.min(a[0], a[2]), aMinX = Math.min(a[1], a[3]), aMaxY = Math.max(a[0], a[2]), aMaxX = Math.max(a[1], a[3]);
        double bMinY = Math.min(b[0], b[2]), bMinX = Math.min(b[1], b[3]), bMaxY = Math.max(b[0], b[2]), bMaxX = Math.max(b[1], b[3]);
        double areaA = (aMaxY - aMinY) * (aMaxX - aMinX), areaB = (bMaxY - bMinY) * (bMaxX - bMinX);
        if (areaA <= 0 || areaB <= 0)
            return 0;
        double inter = Math.max(Math.min(aMaxY, bMaxY) - Math.max(aMinY, bMinY), 0)
                * Math.max(Math.min(aMaxX, bMaxX) - Math.max(aMinX, bMinX), 0);
        return inter / (areaA + areaB - inter);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void manyOverlappingBoxesFollowTheGreedySelection(Nd4jBackend backend) {
        // enough boxes that the CUDA overlap check runs on many blocks, which each wrote the shared selection flag
        Random rng = new Random(53);
        int numBoxes = 600;
        float[][] boxes = new float[numBoxes][];
        float[] scores = new float[numBoxes];
        for (int i = 0; i < numBoxes; i++) {
            float y = rng.nextFloat() * 20, x = rng.nextFloat() * 20;
            float h = 0.5f + rng.nextFloat() * 2, w = 0.5f + rng.nextFloat() * 2;
            boxes[i] = new float[]{y, x, y + h, x + w};
            scores[i] = rng.nextFloat();
        }
        int maxOutputSize = numBoxes;
        double threshold = 0.3;
        int[] expected = greedyReference(boxes, scores, maxOutputSize, threshold);
        INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("non_max_suppression")
                .addInputs(Nd4j.createFromArray(boxes), Nd4j.createFromArray(scores))
                .addIntegerArguments(maxOutputSize).addFloatingPointArguments(threshold).build());
        assertEquals(Nd4j.createFromArray(expected), out[0].castTo(DataType.INT32),
                expected.length + " boxes selected by the reference");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void keepsTheScores(Nd4jBackend backend) {
        // boxes that do not overlap: every one is selected, in decreasing score order
        int numBoxes = 6;
        float[][] boxes = new float[numBoxes][];
        for (int i = 0; i < numBoxes; i++)
            boxes[i] = new float[]{2 * i, 0, 2 * i + 1, 1};
        INDArray boxesArr = Nd4j.createFromArray(boxes);
        INDArray scores = Nd4j.createFromArray(new float[]{0.3f, 0.9f, 0.1f, 0.7f, 0.5f, 0.8f});
        INDArray scoresBefore = scores.dup();
        INDArray expected = Nd4j.createFromArray(new int[]{1, 5, 3, 4, 0, 2});
        for (int run = 0; run < 2; run++) {
            INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("non_max_suppression").addInputs(boxesArr, scores)
                    .addIntegerArguments(numBoxes).addFloatingPointArguments(0.5).build());
            assertTrue(scoresBefore.equalsWithEps(scores, 0.0), "run " + run + ": the scores changed: " + scores);
            assertEquals(expected, out[0].castTo(DataType.INT32), "non_max_suppression run " + run);
        }
    }
}
