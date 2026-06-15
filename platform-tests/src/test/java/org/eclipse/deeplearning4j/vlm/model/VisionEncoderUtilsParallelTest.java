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

package org.eclipse.deeplearning4j.vlm.model;

import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class VisionEncoderUtilsParallelTest extends BaseNd4jTestWithBackends {

    private static final int TARGET_SIZE = 512;

    private BufferedImage createTestFrame(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.BLUE);
        g.fillRect(0, 0, width, height);
        g.dispose();
        return img;
    }

    private Supplier<VLMImagePreprocessor> preprocessorFactory() {
        return () -> {
            org.eclipse.deeplearning4j.llm.config.PreprocessorConfig config =
                    new org.eclipse.deeplearning4j.llm.config.PreprocessorConfig();
            config.setSize(new org.eclipse.deeplearning4j.llm.config.PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
            return VLMImagePreprocessor.fromConfig(config);
        };
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelPreprocess_SingleFrame(Nd4jBackend backend) {
        List<BufferedImage> frames = new ArrayList<>();
        frames.add(createTestFrame(TARGET_SIZE, TARGET_SIZE));

        INDArray result = VisionEncoderUtils.preprocessFramesParallel(
                frames, preprocessorFactory(), TARGET_SIZE, 4);

        assertNotNull(result);
        // Shape should be [1, 1, 3, 512, 512]
        assertEquals(5, result.rank());
        assertEquals(1, result.size(0));
        assertEquals(1, result.size(1));
        assertEquals(3, result.size(2));
        assertEquals(TARGET_SIZE, result.size(3));
        assertEquals(TARGET_SIZE, result.size(4));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelPreprocess_MultipleFrames(Nd4jBackend backend) {
        int numFrames = 5;
        List<BufferedImage> frames = new ArrayList<>();
        for (int i = 0; i < numFrames; i++) {
            frames.add(createTestFrame(TARGET_SIZE, TARGET_SIZE));
        }

        INDArray result = VisionEncoderUtils.preprocessFramesParallel(
                frames, preprocessorFactory(), TARGET_SIZE, 4);

        assertNotNull(result);
        assertEquals(5, result.rank());
        assertEquals(1, result.size(0));
        assertEquals(numFrames, result.size(1));
        assertEquals(3, result.size(2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelPreprocess_EmptyFrames(Nd4jBackend backend) {
        List<BufferedImage> frames = new ArrayList<>();

        INDArray result = VisionEncoderUtils.preprocessFramesParallel(
                frames, preprocessorFactory(), TARGET_SIZE, 4);

        assertNotNull(result);
        assertEquals(0, result.size(1)); // 0 frames
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelPreprocess_SingleThread(Nd4jBackend backend) {
        List<BufferedImage> frames = new ArrayList<>();
        frames.add(createTestFrame(TARGET_SIZE, TARGET_SIZE));
        frames.add(createTestFrame(TARGET_SIZE, TARGET_SIZE));

        // numThreads=1 should fall back to sequential
        INDArray result = VisionEncoderUtils.preprocessFramesParallel(
                frames, preprocessorFactory(), TARGET_SIZE, 1);

        assertNotNull(result);
        assertEquals(2, result.size(1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelMatchesSequential(Nd4jBackend backend) {
        int numFrames = 3;
        List<BufferedImage> frames = new ArrayList<>();
        for (int i = 0; i < numFrames; i++) {
            frames.add(createTestFrame(TARGET_SIZE, TARGET_SIZE));
        }

        // Sequential (1 thread)
        INDArray sequential = VisionEncoderUtils.preprocessFramesParallel(
                frames, preprocessorFactory(), TARGET_SIZE, 1);

        // Parallel (4 threads)
        INDArray parallel = VisionEncoderUtils.preprocessFramesParallel(
                frames, preprocessorFactory(), TARGET_SIZE, 4);

        assertArrayEquals(sequential.shape(), parallel.shape());
        // Values should match — same preprocessing on same images
        double diff = sequential.sub(parallel).ameanNumber().doubleValue();
        assertEquals(0.0, diff, 1e-5, "Parallel and sequential results should match");
    }
}
