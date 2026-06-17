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

package org.eclipse.deeplearning4j.vlm.pipeline;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.*;

/**
 * Orchestrates VLM pipeline parallelism across multiple GPUs.
 *
 * <p>Places vision encoder on one device (e.g., RTX 3070 Ti, 8GB) and decoder
 * on another (e.g., RTX 4090, 24GB). Both GPUs work simultaneously during
 * multi-page batch processing: encoder prepares pages while decoder generates text.</p>
 *
 * <p>For non-P2P GPU pairs, cross-device transfers use host-staged D2H + H2D via
 * {@code CudaAffinityManager.replicateToDevice()}.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * VLMPipelineExecutor pipeline = new VLMPipelineExecutor(
 *     visionEncoder, 1,   // encoder on GPU 1 (3070 Ti)
 *     decoder, 0,          // decoder on GPU 0 (4090)
 *     embedTokens);
 *
 * // Encode all pages on GPU 1, transfer results to GPU 0
 * List<INDArray> embeddings = pipeline.encodeAllPages(pageInputs, outputNames, true);
 *
 * // Decode on GPU 0 (encoder freed, full 8GB available on 3070 Ti)
 * // ... decoder loop on GPU 0 ...
 *
 * pipeline.close();
 * }</pre>
 */
@Slf4j
public class VLMPipelineExecutor implements AutoCloseable {

    private final SameDiff visionEncoder;
    @Getter private final int encoderDeviceId;
    private final SameDiff decoder;
    @Getter private final int decoderDeviceId;
    private final SameDiff embedTokens;

    /** Single-thread executor for encoder — ensures all encoder work stays on encoderDeviceId. */
    private final ExecutorService encoderExecutor;

    private volatile boolean closed = false;
    private volatile boolean encoderFreed = false;

    /**
     * Create a VLM pipeline executor for multi-GPU inference.
     *
     * @param visionEncoder   the vision encoder SameDiff graph
     * @param encoderDeviceId CUDA device ID for the encoder (e.g., 1 for RTX 3070 Ti)
     * @param decoder         the decoder SameDiff graph
     * @param decoderDeviceId CUDA device ID for the decoder (e.g., 0 for RTX 4090)
     * @param embedTokens     the token embedding graph (may be null if decoder handles it)
     */
    public VLMPipelineExecutor(SameDiff visionEncoder, int encoderDeviceId,
                                SameDiff decoder, int decoderDeviceId,
                                SameDiff embedTokens) {
        this.visionEncoder = visionEncoder;
        this.encoderDeviceId = encoderDeviceId;
        this.decoder = decoder;
        this.decoderDeviceId = decoderDeviceId;
        this.embedTokens = embedTokens;

        // Create encoder thread with device affinity
        this.encoderExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(() -> {
                DeviceMemoryManager.getInstance().switchDevice(encoderDeviceId, "VLMPipelineExecutor", "encoder-thread");
                r.run();
            }, "VLMPipeline-Encoder-GPU" + encoderDeviceId);
            t.setDaemon(true);
            return t;
        });

        log.info("VLMPipelineExecutor: encoder on GPU {}, decoder on GPU {}",
                encoderDeviceId, decoderDeviceId);
    }

    /**
     * Submit vision encoding asynchronously on the encoder device.
     * The returned future resolves to embeddings already transferred to the decoder device.
     *
     * @param visionInputs placeholder inputs for the vision encoder (e.g., pixel_values)
     * @param outputNames  output variable names to retrieve
     * @return future resolving to vision embeddings on the decoder device
     */
    public Future<INDArray> encodeAsync(Map<String, INDArray> visionInputs, String[] outputNames) {
        checkNotClosed();
        if (encoderFreed) {
            throw new IllegalStateException("Vision encoder has been freed");
        }

        return encoderExecutor.submit(() -> {
            // Execute encoder on encoder device (thread already affined)
            Map<String, INDArray> outputs = visionEncoder.output(visionInputs, outputNames);

            // Pick the first available output
            INDArray embeddings = null;
            for (String name : outputNames) {
                embeddings = outputs.get(name);
                if (embeddings != null) break;
            }
            if (embeddings == null) {
                throw new IllegalStateException("No vision encoder output found for: " + Arrays.toString(outputNames));
            }

            // Transfer to decoder device (host-staged for non-P2P)
            if (encoderDeviceId != decoderDeviceId) {
                INDArray transferred = Nd4j.getAffinityManager().replicateToDevice(decoderDeviceId, embeddings);
                // Close the encoder-device copy (not needed after transfer)
                if (embeddings.data() != null && !embeddings.data().isConstant() && embeddings.closeable()) {
                    embeddings.close();
                }
                embeddings = transferred;
            }

            return embeddings;
        });
    }

    /**
     * Encode all pages sequentially on the encoder device, returning embeddings on the decoder device.
     * Optionally frees the vision encoder after all pages are processed.
     *
     * @param pageInputs   list of placeholder input maps, one per page
     * @param outputNames  output variable names to retrieve
     * @param freeEncoder  if true, free the vision encoder after encoding all pages
     * @return list of vision embeddings, each on the decoder device
     */
    public List<INDArray> encodeAllPages(List<Map<String, INDArray>> pageInputs,
                                          String[] outputNames, boolean freeEncoder) {
        checkNotClosed();
        if (encoderFreed) {
            throw new IllegalStateException("Vision encoder has been freed");
        }

        List<INDArray> results = new ArrayList<>(pageInputs.size());
        for (int i = 0; i < pageInputs.size(); i++) {
            Map<String, INDArray> inputs = pageInputs.get(i);
            try {
                Future<INDArray> future = encodeAsync(inputs, outputNames);
                INDArray embeddings = future.get(5, TimeUnit.MINUTES);
                results.add(embeddings);
                log.info("VLMPipeline: encoded page {}/{}, embeddings shape={}",
                        i + 1, pageInputs.size(), Arrays.toString(embeddings.shape()));
            } catch (TimeoutException e) {
                throw new RuntimeException("Vision encoding timed out for page " + i, e);
            } catch (ExecutionException e) {
                throw new RuntimeException("Vision encoding failed for page " + i, e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Vision encoding interrupted for page " + i, e);
            }
        }

        if (freeEncoder) {
            freeVisionEncoder();
        }

        return results;
    }

    /**
     * Encode all pages with pipeline overlap: submit page N+1 preprocessing while
     * encoding page N. Returns embeddings on the decoder device.
     *
     * @param pageInputSupplier supplies placeholder input maps lazily (preprocessing)
     * @param numPages          total number of pages
     * @param outputNames       output variable names to retrieve
     * @param freeEncoder       if true, free the vision encoder after encoding all pages
     * @return list of vision embeddings, each on the decoder device
     */
    public List<INDArray> encodeAllPagesPipelined(
            java.util.function.IntFunction<Map<String, INDArray>> pageInputSupplier,
            int numPages, String[] outputNames, boolean freeEncoder) {
        checkNotClosed();
        if (encoderFreed) {
            throw new IllegalStateException("Vision encoder has been freed");
        }

        List<INDArray> results = new ArrayList<>(numPages);

        // Submit first page
        Future<INDArray> currentFuture = encodeAsync(pageInputSupplier.apply(0), outputNames);

        for (int i = 0; i < numPages; i++) {
            // Submit next page while current encodes (pipeline overlap)
            Future<INDArray> nextFuture = null;
            if (i + 1 < numPages) {
                Map<String, INDArray> nextInputs = pageInputSupplier.apply(i + 1);
                nextFuture = encodeAsync(nextInputs, outputNames);
            }

            // Wait for current page
            try {
                INDArray embeddings = currentFuture.get(5, TimeUnit.MINUTES);
                results.add(embeddings);
                log.info("VLMPipeline: encoded page {}/{}", i + 1, numPages);
            } catch (TimeoutException e) {
                throw new RuntimeException("Vision encoding timed out for page " + i, e);
            } catch (ExecutionException e) {
                throw new RuntimeException("Vision encoding failed for page " + i, e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Vision encoding interrupted for page " + i, e);
            }

            currentFuture = nextFuture;
        }

        if (freeEncoder) {
            freeVisionEncoder();
        }

        return results;
    }

    /**
     * Free the vision encoder constants to recover GPU memory on the encoder device.
     * After calling this, no further encoding is possible.
     */
    public void freeVisionEncoder() {
        if (encoderFreed) return;
        encoderFreed = true;

        // Run on encoder thread to ensure correct device context
        try {
            encoderExecutor.submit(() -> {
                log.info("VLMPipeline: freeing vision encoder on GPU {}", encoderDeviceId);
                try {
                    // Reset session to clear session references
                    visionEncoder.resetSession();
                    // Free constant arrays (model weights)
                    Map<String, INDArray> constants = new HashMap<>();
                    for (org.nd4j.autodiff.samediff.SDVariable sdVar : visionEncoder.variables()) {
                        if (sdVar != null && sdVar.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT) {
                            INDArray arr = sdVar.getArr();
                            if (arr != null) {
                                constants.put(sdVar.name(), arr);
                            }
                        }
                    }
                    for (Map.Entry<String, INDArray> entry : constants.entrySet()) {
                        INDArray arr = entry.getValue();
                        if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                            arr.data().setConstant(false);
                            try { arr.close(); } catch (Exception ignored) {}
                        }
                    }
                    log.info("VLMPipeline: freed {} vision encoder constants", constants.size());
                } catch (Exception e) {
                    log.warn("Error freeing vision encoder: {}", e.getMessage());
                }
            }).get(30, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.warn("Timeout/error freeing vision encoder: {}", e.getMessage());
        }
    }

    /**
     * @return whether the vision encoder has been freed
     */
    public boolean isEncoderFreed() {
        return encoderFreed;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("VLMPipelineExecutor has been closed");
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (!encoderFreed) {
            freeVisionEncoder();
        }

        encoderExecutor.shutdown();
        try {
            if (!encoderExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                encoderExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            encoderExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        log.info("VLMPipelineExecutor closed");
    }
}
