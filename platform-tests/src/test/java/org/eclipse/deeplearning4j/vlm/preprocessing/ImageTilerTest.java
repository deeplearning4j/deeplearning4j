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

package org.eclipse.deeplearning4j.vlm.preprocessing;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class ImageTilerTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplitSmallImage(Nd4jBackend backend) {
        // Image smaller than maxSize → 1 frame (global only), no tiling
        BufferedImage small = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLM(small, 512);

        assertEquals(1, result.getTotalFrames()); // only global
        assertEquals(0, result.numRows);
        assertEquals(0, result.numCols);
        assertEquals(0, result.getTileCount());

        // The global frame should be maxSize x maxSize
        BufferedImage globalFrame = result.frames.get(0);
        assertEquals(512, globalFrame.getWidth());
        assertEquals(512, globalFrame.getHeight());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplitLargeImage(Nd4jBackend backend) {
        // 1024x1024 with maxSize=512 → 2x2 grid (4 tiles) + 1 global = 5 frames
        BufferedImage large = new BufferedImage(1024, 1024, BufferedImage.TYPE_INT_RGB);
        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLM(large, 512);

        assertEquals(2, result.numRows);
        assertEquals(2, result.numCols);
        assertEquals(4, result.getTileCount());
        assertEquals(5, result.getTotalFrames()); // 4 tiles + 1 global

        // All frames should be maxSize x maxSize
        for (BufferedImage frame : result.frames) {
            assertEquals(512, frame.getWidth());
            assertEquals(512, frame.getHeight());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplitWithMaxTilesLimit(Nd4jBackend backend) {
        // Large image with maxTiles=4 should cap the grid
        BufferedImage large = new BufferedImage(2048, 2048, BufferedImage.TYPE_INT_RGB);
        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLM(large, 512, 4);

        int tileCount = result.getTileCount();
        assertTrue(tileCount <= 4, "Expected <= 4 tiles, got " + tileCount);
        // Total frames = tiles + 1 global
        assertEquals(tileCount + 1, result.getTotalFrames());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplitWithMaxTilesOne(Nd4jBackend backend) {
        // maxTiles=1 → skip tiling, global only
        BufferedImage large = new BufferedImage(1024, 1024, BufferedImage.TYPE_INT_RGB);
        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLM(large, 512, 1);

        assertEquals(0, result.numRows);
        assertEquals(0, result.numCols);
        assertEquals(1, result.getTotalFrames());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplitRectangularImageUsesAspectPreservingPadding(Nd4jBackend backend) {
        BufferedImage rectangular = new BufferedImage(800, 400, BufferedImage.TYPE_INT_RGB);
        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLM(rectangular, 512);

        assertEquals(1, result.numRows);
        assertEquals(2, result.numCols);
        assertEquals(3, result.getTotalFrames());

        for (BufferedImage frame : result.frames) {
            assertEquals(512, frame.getWidth());
            assertEquals(512, frame.getHeight());
        }

        assertEquals(400, result.contentRegions.get(0).width);
        assertEquals(400, result.contentRegions.get(0).height);
        assertEquals(400, result.contentRegions.get(1).width);
        assertEquals(400, result.contentRegions.get(1).height);
        assertEquals(512, result.contentRegions.get(2).width);
        assertEquals(256, result.contentRegions.get(2).height);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResizeLongestEdge(Nd4jBackend backend) {
        BufferedImage image = new BufferedImage(800, 400, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = ImageTiler.resizeLongestEdge(image, 200);

        // Longest edge (800) should become 200, aspect ratio preserved
        assertEquals(200, resized.getWidth());
        assertEquals(100, resized.getHeight());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResizeLongestEdgeAlreadyFits(Nd4jBackend backend) {
        BufferedImage image = new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = ImageTiler.resizeLongestEdge(image, 200);

        // Already matches target, should return same image
        assertSame(image, resized);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreatePixelAttentionMask(Nd4jBackend backend) {
        // Full content → all ones
        INDArray fullMask = ImageTiler.createPixelAttentionMask(512, 512, 512);
        assertArrayEquals(new long[]{1, 512, 512}, fullMask.shape());
        assertEquals(DataType.BOOL, fullMask.dataType());
        // Cast BOOL to INT32 for sum (BOOL doesn't support sum directly)
        assertEquals(512L * 512L, fullMask.castTo(DataType.INT32).sumNumber().longValue());

        // Partial content → ones in content area, zeros in padding
        INDArray partialMask = ImageTiler.createPixelAttentionMask(300, 200, 512);
        assertArrayEquals(new long[]{1, 512, 512}, partialMask.shape());

        // Content area (200 rows x 300 cols) should be 1
        long onesCount = partialMask.castTo(DataType.INT32).sumNumber().longValue();
        assertEquals(200L * 300L, onesCount);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResizeToFit(Nd4jBackend backend) {
        BufferedImage image = new BufferedImage(800, 400, BufferedImage.TYPE_INT_RGB);
        ImageTiler.ResizeResult result = ImageTiler.resizeToFit(image, 400, 400);

        // Width limited: 800→400, height proportionally: 400→200
        assertEquals(400, result.width);
        assertEquals(200, result.height);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResizeToFitAlreadyFits(Nd4jBackend backend) {
        BufferedImage image = new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB);
        ImageTiler.ResizeResult result = ImageTiler.resizeToFit(image, 400, 400);

        // Already fits, should return same image
        assertSame(image, result.image);
        assertEquals(200, result.width);
        assertEquals(100, result.height);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPadToSize(Nd4jBackend backend) {
        BufferedImage image = new BufferedImage(300, 200, BufferedImage.TYPE_INT_RGB);
        BufferedImage padded = ImageTiler.padToSize(image, 512, 512);

        assertEquals(512, padded.getWidth());
        assertEquals(512, padded.getHeight());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPadToSizeAlreadyCorrect(Nd4jBackend backend) {
        BufferedImage image = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        BufferedImage padded = ImageTiler.padToSize(image, 512, 512);

        assertSame(image, padded);
    }
}
