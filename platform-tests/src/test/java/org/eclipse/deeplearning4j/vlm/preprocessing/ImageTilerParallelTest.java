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

import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.awt.*;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class ImageTilerParallelTest extends BaseNd4jTestWithBackends {

    private BufferedImage createTestImage(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        // Fill with a gradient so tiles have distinguishable content
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (x * 255) / Math.max(width - 1, 1);
                int gVal = (y * 255) / Math.max(height - 1, 1);
                img.setRGB(x, y, new Color(r, gVal, 128).getRGB());
            }
        }
        g.dispose();
        return img;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelMatchesSequential_SmallImage(Nd4jBackend backend) {
        BufferedImage img = createTestImage(256, 256);

        ImageTiler.SplitImageResult sequential = ImageTiler.splitImageForVLM(img, 512);
        ImageTiler.SplitImageResult parallel = ImageTiler.splitImageForVLMParallel(img, 512, 9, 4);

        assertEquals(sequential.getTotalFrames(), parallel.getTotalFrames());
        assertEquals(sequential.numRows, parallel.numRows);
        assertEquals(sequential.numCols, parallel.numCols);
        assertEquals(sequential.getTileCount(), parallel.getTileCount());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelMatchesSequential_LargeImage(Nd4jBackend backend) {
        BufferedImage img = createTestImage(1024, 1024);

        ImageTiler.SplitImageResult sequential = ImageTiler.splitImageForVLM(img, 512);
        ImageTiler.SplitImageResult parallel = ImageTiler.splitImageForVLMParallel(img, 512, 9, 4);

        assertEquals(sequential.getTotalFrames(), parallel.getTotalFrames());
        assertEquals(sequential.numRows, parallel.numRows);
        assertEquals(sequential.numCols, parallel.numCols);

        // All frames should have correct dimensions
        for (int i = 0; i < parallel.getTotalFrames(); i++) {
            assertEquals(512, parallel.frames.get(i).getWidth());
            assertEquals(512, parallel.frames.get(i).getHeight());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelMatchesSequential_MaxTilesLimit(Nd4jBackend backend) {
        BufferedImage img = createTestImage(2048, 2048);

        ImageTiler.SplitImageResult sequential = ImageTiler.splitImageForVLM(img, 512, 4);
        ImageTiler.SplitImageResult parallel = ImageTiler.splitImageForVLMParallel(img, 512, 4, 4);

        assertEquals(sequential.getTotalFrames(), parallel.getTotalFrames());
        assertTrue(parallel.getTileCount() <= 4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallel_SingleThread_FallsBack(Nd4jBackend backend) {
        BufferedImage img = createTestImage(1024, 1024);

        // numThreads=1 should work fine (single thread path)
        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLMParallel(img, 512, 9, 1);
        assertTrue(result.getTotalFrames() > 0);

        // Compare with sequential
        ImageTiler.SplitImageResult sequential = ImageTiler.splitImageForVLM(img, 512);
        assertEquals(sequential.getTotalFrames(), result.getTotalFrames());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallel_ContentRegions(Nd4jBackend backend) {
        BufferedImage img = createTestImage(1024, 512);

        ImageTiler.SplitImageResult result = ImageTiler.splitImageForVLMParallel(img, 512, 9, 4);

        // Should have content region for each frame
        assertEquals(result.getTotalFrames(), result.contentRegions.size());

        for (ImageTiler.ContentRegion region : result.contentRegions) {
            assertTrue(region.width > 0);
            assertTrue(region.height > 0);
            assertTrue(region.width <= 512);
            assertTrue(region.height <= 512);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallel_RectangularImage(Nd4jBackend backend) {
        BufferedImage img = createTestImage(800, 400);

        ImageTiler.SplitImageResult sequential = ImageTiler.splitImageForVLM(img, 512);
        ImageTiler.SplitImageResult parallel = ImageTiler.splitImageForVLMParallel(img, 512, 9, 4);

        assertEquals(sequential.getTotalFrames(), parallel.getTotalFrames());
        assertEquals(sequential.numRows, parallel.numRows);
        assertEquals(sequential.numCols, parallel.numCols);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelMatchesSequential_ContentRegions_RectangularImage(Nd4jBackend backend) {
        BufferedImage img = createTestImage(800, 400);

        ImageTiler.SplitImageResult sequential = ImageTiler.splitImageForVLM(img, 512);
        ImageTiler.SplitImageResult parallel = ImageTiler.splitImageForVLMParallel(img, 512, 9, 4);

        assertEquals(sequential.contentRegions.size(), parallel.contentRegions.size());
        // Tiles-first, global-last ordering. Global is at the last index in both.
        int lastIdx = sequential.contentRegions.size() - 1;
        ImageTiler.ContentRegion seqGlobal = sequential.contentRegions.get(lastIdx);
        ImageTiler.ContentRegion parGlobal = parallel.contentRegions.get(lastIdx);
        assertEquals(seqGlobal.width, parGlobal.width, "global content width mismatch");
        assertEquals(seqGlobal.height, parGlobal.height, "global content height mismatch");

        // Tile content regions may differ because parallel pre-resizes the source
        // to exact tile multiples while sequential crops at original resolution.
        // Just verify all content regions are valid (positive dimensions, <= maxSize).
        for (int i = 0; i < lastIdx; i++) {
            ImageTiler.ContentRegion region = parallel.contentRegions.get(i);
            assertTrue(region.width > 0, "tile " + i + " content width must be positive");
            assertTrue(region.height > 0, "tile " + i + " content height must be positive");
            assertTrue(region.width <= 512, "tile " + i + " content width must be <= maxSize");
            assertTrue(region.height <= 512, "tile " + i + " content height must be <= maxSize");
        }
    }
}
