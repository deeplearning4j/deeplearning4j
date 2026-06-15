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

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Utilities for working with vision encoder outputs and preprocessing frames.
 */
@Slf4j
public class VisionEncoderUtils {

    public static class VisionOutput {
        public final String name;
        public final INDArray tensor;

        public VisionOutput(String name, INDArray tensor) {
            this.name = name;
            this.tensor = tensor;
        }
    }

    /**
     * Select the best vision output tensor from encoder outputs.
     * Prefers last_hidden_state, falls back to the rank-3 tensor with largest sequence dimension.
     *
     * @param outputs the vision encoder output map
     * @return the selected VisionOutput, or null if no usable output found
     */
    public static VisionOutput selectVisionOutput(Map<String, INDArray> outputs) {
        if (outputs == null || outputs.isEmpty()) {
            return null;
        }

        for (var entry : outputs.entrySet()) {
            String name = entry.getKey();
            INDArray tensor = entry.getValue();
            if (name.contains("last_hidden_state") && tensor != null && tensor.rank() == 3) {
                return new VisionOutput(name, tensor);
            }
        }

        VisionOutput best = null;
        for (var entry : outputs.entrySet()) {
            INDArray tensor = entry.getValue();
            if (tensor != null && tensor.rank() == 3) {
                if (best == null || tensor.size(1) > best.tensor.size(1)) {
                    best = new VisionOutput(entry.getKey(), tensor);
                }
            }
        }

        return best;
    }

    /**
     * Preprocess multiple image frames into a 5D tensor [batch, numFrames, channels, H, W].
     *
     * @param frames list of BufferedImage frames
     * @param preprocessor the VLM preprocessor to use
     * @param targetSize the target size for each frame (e.g. 512)
     * @return INDArray with shape [1, numFrames, 3, targetSize, targetSize]
     */
    public static INDArray preprocessFrames(List<BufferedImage> frames,
                                            VLMImagePreprocessor preprocessor,
                                            int targetSize) {
        int numFrames = frames.size();
        INDArray result = Nd4j.create(DataType.FLOAT, 1, numFrames, 3, targetSize, targetSize);

        long totalStartTime = System.currentTimeMillis();
        for (int f = 0; f < numFrames; f++) {
            BufferedImage frame = frames.get(f);

            INDArray frameTensor = preprocessor.preprocess(frame);  // [1, 3, H, W]

            INDArray targetSlice = result.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(f),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            INDArray sourceSlice = frameTensor.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            targetSlice.assign(sourceSlice);

            frameTensor.close();

            log.debug("Preprocessed frame {}/{}: min={}, max={}, mean={}",
                    f + 1, numFrames,
                    targetSlice.minNumber(), targetSlice.maxNumber(), targetSlice.meanNumber());
        }

        long totalTime = System.currentTimeMillis() - totalStartTime;
        log.info("Preprocessed {} frames in {} ms ({} ms/frame avg)",
                numFrames, totalTime, numFrames > 0 ? totalTime / numFrames : 0);

        return result;
    }

    /**
     * Preprocess multiple image frames in parallel using a thread pool.
     * Each frame's CPU preprocessing (resize, normalize, convert) runs concurrently.
     * The results are assembled into a 5D tensor after all threads complete.
     *
     * @param frames list of BufferedImage frames
     * @param preprocessorFactory creates a VLMImagePreprocessor per thread (not shared — they are not thread-safe)
     * @param targetSize the target size for each frame (e.g. 512)
     * @param numThreads number of parallel preprocessing threads
     * @return INDArray with shape [1, numFrames, 3, targetSize, targetSize]
     */
    public static INDArray preprocessFramesParallel(List<BufferedImage> frames,
                                                    java.util.function.Supplier<VLMImagePreprocessor> preprocessorFactory,
                                                    int targetSize, int numThreads) {
        int numFrames = frames.size();
        if (numFrames == 0) {
            return Nd4j.create(DataType.FLOAT, 1, 0, 3, targetSize, targetSize);
        }

        int threads = Math.min(numThreads, numFrames);
        INDArray[] frameTensors = new INDArray[numFrames];

        long totalStartTime = System.currentTimeMillis();

        if (threads <= 1) {
            // Fall back to sequential for single thread
            VLMImagePreprocessor preprocessor = preprocessorFactory.get();
            for (int f = 0; f < numFrames; f++) {
                frameTensors[f] = preprocessor.preprocess(frames.get(f));
            }
            preprocessor.shutdown();
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(threads, r -> {
                Thread t = new Thread(r, "VisionPreprocess");
                t.setDaemon(true);
                return t;
            });
            try {
                List<Future<?>> futures = new ArrayList<>(numFrames);
                // Each thread gets its own preprocessor (not thread-safe)
                ThreadLocal<VLMImagePreprocessor> threadPreprocessor = ThreadLocal.withInitial(preprocessorFactory::get);
                for (int f = 0; f < numFrames; f++) {
                    final int idx = f;
                    futures.add(pool.submit(() -> {
                        frameTensors[idx] = threadPreprocessor.get().preprocess(frames.get(idx));
                    }));
                }
                for (Future<?> future : futures) {
                    future.get();
                }
            } catch (Exception e) {
                throw new RuntimeException("Parallel frame preprocessing failed", e);
            } finally {
                pool.shutdown();
            }
        }

        // Assemble into 5D tensor
        INDArray result = Nd4j.create(DataType.FLOAT, 1, numFrames, 3, targetSize, targetSize);
        for (int f = 0; f < numFrames; f++) {
            INDArray targetSlice = result.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(f),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            INDArray sourceSlice = frameTensors[f].get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            targetSlice.assign(sourceSlice);
            frameTensors[f].close();
        }

        long totalTime = System.currentTimeMillis() - totalStartTime;
        log.info("Parallel-preprocessed {} frames in {} ms ({} ms/frame avg, {} threads)",
                numFrames, totalTime, numFrames > 0 ? totalTime / numFrames : 0, threads);

        return result;
    }

    /**
     * Check if an array contains all zeros or NaN/Inf values.
     *
     * @param arr the array to check
     * @return true if all zeros, NaN, or Inf
     */
    public static boolean isAllZeroOrNaN(INDArray arr) {
        double min = arr.minNumber().doubleValue();
        double max = arr.maxNumber().doubleValue();
        if (Double.isNaN(min) || Double.isNaN(max) || Double.isInfinite(min) || Double.isInfinite(max)) {
            return true;
        }
        return min == 0.0 && max == 0.0;
    }
}
