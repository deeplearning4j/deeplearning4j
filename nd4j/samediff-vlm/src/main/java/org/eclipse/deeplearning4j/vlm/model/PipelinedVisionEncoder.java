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
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.bytedeco.javacpp.LongPointer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Supplier;

/**
 * Pipelined vision encoder that overlaps CPU preprocessing with GPU encoding.
 *
 * <p>While the GPU is encoding frames from page N, the CPU preprocesses frames
 * from page N+1. This hides preprocessing latency behind GPU compute time.</p>
 *
 * <p>For K pages with F frames each:</p>
 * <ul>
 *   <li>Sequential: preprocess(F) + encode(F) for each page = K * (preprocess + encode)</li>
 *   <li>Pipelined: preprocess(page0) + [encode(pageN) || preprocess(pageN+1)] * (K-1) + encode(pageK-1)</li>
 *   <li>Speedup: ~preprocess_time * (K-1) saved</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class PipelinedVisionEncoder {

    private final SameDiff visionEncoder;
    private final String[] outputNames;
    private final boolean hasMaskInput;
    private final int visionChunkSize;
    private final ExecutorService preprocessPool;

    /**
     * Create a pipelined vision encoder.
     *
     * @param visionEncoder the vision encoder SameDiff model
     * @param visionChunkSize number of frames per encoder forward pass
     * @param preprocessThreads number of threads for CPU preprocessing
     */
    public PipelinedVisionEncoder(SameDiff visionEncoder, int visionChunkSize, int preprocessThreads) {
        this.visionEncoder = visionEncoder;
        this.outputNames = visionEncoder.outputs().toArray(new String[0]);
        this.hasMaskInput = visionEncoder.getVariable("pixel_attention_mask") != null;
        this.visionChunkSize = visionChunkSize;
        this.preprocessPool = Executors.newFixedThreadPool(preprocessThreads, r -> {
            Thread t = new Thread(r, "VisionPreprocess-Pipeline");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * Encode multiple pages with pipelined preprocessing.
     *
     * <p>Preprocesses page N+1's frames on CPU while page N's frames are being
     * encoded on GPU. Returns per-page embeddings concatenated along the
     * sequence dimension.</p>
     *
     * @param pageSplitResults per-page tiling results containing frames
     * @param preprocessorFactory creates a VLMImagePreprocessor per thread
     * @param targetSize target frame size (e.g. 512)
     * @return list of per-page vision embeddings, each shape [1, pageSeqLen, hidden]
     */
    public List<INDArray> encodePipelined(
            List<ImageTiler.SplitImageResult> pageSplitResults,
            Supplier<VLMImagePreprocessor> preprocessorFactory,
            int targetSize) {

        int numPages = pageSplitResults.size();
        List<INDArray> pageEmbeddings = new ArrayList<>(numPages);

        // Start preprocessing first page immediately
        Future<PreprocessedPage> nextPageFuture = submitPreprocess(
                pageSplitResults.get(0), preprocessorFactory, targetSize, 0);

        for (int pageIdx = 0; pageIdx < numPages; pageIdx++) {
            long pageStart = System.currentTimeMillis();

            // Wait for current page's preprocessing
            PreprocessedPage currentPage;
            try {
                currentPage = nextPageFuture.get();
            } catch (Exception e) {
                throw new RuntimeException("Failed to preprocess page " + pageIdx, e);
            }

            // Start preprocessing NEXT page while we encode current page
            if (pageIdx + 1 < numPages) {
                nextPageFuture = submitPreprocess(
                        pageSplitResults.get(pageIdx + 1), preprocessorFactory, targetSize, pageIdx + 1);
            }

            // Encode current page's frames in chunks
            List<INDArray> frameEmbeddings = encodeFrames(currentPage);

            // Concatenate frame embeddings: [1, totalTokens, hidden]
            INDArray pageEmbedding;
            if (frameEmbeddings.size() == 1) {
                pageEmbedding = frameEmbeddings.get(0);
            } else {
                pageEmbedding = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
                for (INDArray fe : frameEmbeddings) {
                    if (fe != null && !fe.wasClosed()) fe.close();
                }
            }

            // Close preprocessed frame tensors
            currentPage.close();

            // Trim GPU memory pools on all devices to release reserved-but-unused memory
            // back to the driver before the next page starts encoding. Without this,
            // the pool retains all reserved memory from this page's encoding, and the
            // next page's allocations fail with OOM despite the memory being logically free.
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }

            pageEmbeddings.add(pageEmbedding);
            long pageTime = System.currentTimeMillis() - pageStart;
            log.info("  Page {}: {} frames -> embedding shape={} [{}ms]",
                    pageIdx, currentPage.numFrames,
                    java.util.Arrays.toString(pageEmbedding.shape()), pageTime);
        }

        return pageEmbeddings;
    }

    private Future<PreprocessedPage> submitPreprocess(
            ImageTiler.SplitImageResult splitResult,
            Supplier<VLMImagePreprocessor> preprocessorFactory,
            int targetSize, int pageIdx) {

        return preprocessPool.submit(() -> {
            long start = System.currentTimeMillis();
            int numFrames = splitResult.getTotalFrames();
            INDArray[] frameTensors = new INDArray[numFrames];
            INDArray[] masks = new INDArray[numFrames];

            VLMImagePreprocessor preprocessor = preprocessorFactory.get();
            for (int f = 0; f < numFrames; f++) {
                INDArray tensor = preprocessor.preprocess(splitResult.frames.get(f));
                frameTensors[f] = tensor.reshape(1, 1, 3, targetSize, targetSize);

                ImageTiler.ContentRegion region = splitResult.contentRegions.get(f);
                masks[f] = ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize);
            }
            preprocessor.shutdown();

            log.debug("  Preprocessed page {} ({} frames) in {}ms",
                    pageIdx, numFrames, System.currentTimeMillis() - start);

            return new PreprocessedPage(frameTensors, masks, numFrames);
        });
    }

    private List<INDArray> encodeFrames(PreprocessedPage page) {
        List<INDArray> embeddings = new ArrayList<>();
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        // Track per-device pool stats across frames to detect memory growth
        long[] prevPoolUsed = new long[numDevices];
        long[] prevPoolReserved = new long[numDevices];

        // Process frames one at a time through the vision encoder. ONNX vision
        // encoder models (SigLIP etc.) only support batch_size=1 — internal reshape
        // and attention ops break with batch_size>1 (CUDA error 700).
        for (int f = 0; f < page.numFrames; f++) {
            // Pre-flight memory check: trim pools and verify sufficient memory BEFORE
            // starting a frame that takes ~30s. Without this, OOM occurs at the end of
            // a frame after wasting the entire encode time.
            if (f > 0) {
                for (int d = 0; d < numDevices; d++) {
                    nativeOps.trimMemoryPool(d);
                }
            }

            // Log memory state before each frame so OOM can be diagnosed from logs
            for (int d = 0; d < numDevices; d++) {
                long freeMem = nativeOps.getDeviceFreeMemory(d);
                long totalMem = nativeOps.getDeviceTotalMemory(d);
                long usedMem = totalMem - freeMem;
                double pctUsed = 100.0 * usedMem / totalMem;
                log.info("  Frame {}/{}: GPU {} memory: {}/{} MB ({}% used)",
                        f, page.numFrames, d, usedMem / (1024 * 1024),
                        totalMem / (1024 * 1024), String.format("%.1f", pctUsed));

                // Pool stats: poolUsed = live allocations, poolReserved = total held by pool
                // If poolUsed grows between frames, memory is leaking (not just pool-reserved)
                try {
                    LongPointer usedPtr = new LongPointer(1);
                    LongPointer reservedPtr = new LongPointer(1);
                    nativeOps.getMemoryPoolStats(d, usedPtr, reservedPtr);
                    long poolUsedMB = usedPtr.get() / (1024 * 1024);
                    long poolReservedMB = reservedPtr.get() / (1024 * 1024);
                    long deltaUsedMB = (f > 0) ? (usedPtr.get() - prevPoolUsed[d]) / (1024 * 1024) : 0;
                    long deltaReservedMB = (f > 0) ? (reservedPtr.get() - prevPoolReserved[d]) / (1024 * 1024) : 0;
                    if (f > 0) {
                        log.info("  Frame {}/{}: GPU {} pool: used={}MB ({}{}MB), reserved={}MB ({}{}MB)",
                                f, page.numFrames, d,
                                poolUsedMB, deltaUsedMB >= 0 ? "+" : "", deltaUsedMB,
                                poolReservedMB, deltaReservedMB >= 0 ? "+" : "", deltaReservedMB);
                    } else {
                        log.info("  Frame {}/{}: GPU {} pool: used={}MB, reserved={}MB",
                                f, page.numFrames, d, poolUsedMB, poolReservedMB);
                    }
                    prevPoolUsed[d] = usedPtr.get();
                    prevPoolReserved[d] = reservedPtr.get();
                } catch (Exception e) {
                    // CPU backend — pool stats not available
                }

                // Fail fast if GPU is critically low (< 5% free = likely OOM during frame)
                if (pctUsed > 95.0) {
                    log.warn("  GPU {} is critically low on memory ({}% used). "
                            + "Frame {} may OOM.", d, String.format("%.1f", pctUsed), f);
                }
            }

            INDArray pixelValues = page.frameTensors[f];
            INDArray mask = page.masks[f];

            Map<String, INDArray> inputs = new java.util.HashMap<>();
            inputs.put("pixel_values", pixelValues);
            if (hasMaskInput) {
                inputs.put("pixel_attention_mask", mask);
            }

            Map<String, INDArray> outputs = visionEncoder.output(inputs, outputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(outputs);
            if (selected == null) {
                throw new RuntimeException("Vision encoder produced no output for frame " + f);
            }

            embeddings.add(selected.tensor.dup());

            for (var entry : outputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            // directExecHelper poisons placeholders via setCloseable(false) → setConstant(true).
            // Must undo before close, otherwise close() is a no-op → leaked to GC → SIGABRT.
            pixelValues.setCloseable(true);
            mask.setCloseable(true);
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();
        }

        return embeddings;
    }

    /**
     * Shutdown the preprocessing thread pool.
     */
    public void shutdown() {
        preprocessPool.shutdown();
        try {
            if (!preprocessPool.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                preprocessPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            preprocessPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    private static class PreprocessedPage {
        final INDArray[] frameTensors;
        final INDArray[] masks;
        final int numFrames;

        PreprocessedPage(INDArray[] frameTensors, INDArray[] masks, int numFrames) {
            this.frameTensors = frameTensors;
            this.masks = masks;
            this.numFrames = numFrames;
        }

        void close() {
            for (INDArray t : frameTensors) {
                if (t != null && !t.wasClosed()) { t.setCloseable(true); t.close(); }
            }
            for (INDArray m : masks) {
                if (m != null && !m.wasClosed()) { m.setCloseable(true); m.close(); }
            }
        }
    }
}
