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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Image tiling utilities for Vision-Language Models.
 *
 * Splits an image into a grid of tiles plus a global frame for VLM processing.
 *
 * <p>Each cropped tile is resized to fit within {@code maxSize x maxSize} while preserving
 * aspect ratio, then padded out to the model's square input size. The returned
 * {@link ContentRegion} values capture the unpadded content size for each frame so downstream
 * pixel-attention masks reflect the true occupied area instead of treating every frame as a
 * fully populated square.</p>
 */
@Slf4j
public class ImageTiler {

    public static class SplitImageResult {
        public final List<BufferedImage> frames;
        public final List<ContentRegion> contentRegions;
        public final int numRows;
        public final int numCols;

        public SplitImageResult(List<BufferedImage> frames, List<ContentRegion> contentRegions, int numRows, int numCols) {
            this.frames = frames;
            this.contentRegions = contentRegions;
            this.numRows = numRows;
            this.numCols = numCols;
        }

        public int getTileCount() {
            return numRows * numCols;
        }

        public int getTotalFrames() {
            return frames.size();
        }
    }

    public static class ContentRegion {
        public final int width;
        public final int height;

        public ContentRegion(int width, int height) {
            this.width = width;
            this.height = height;
        }
    }

    public static class ResizeResult {
        public final BufferedImage image;
        public final int width;
        public final int height;

        public ResizeResult(BufferedImage image, int width, int height) {
            this.image = image;
            this.width = width;
            this.height = height;
        }
    }

    /**
     * Split an image into tiles for VLM processing (e.g. SmolDocling/Idefics3).
     *
     * The algorithm:
     * 1. Choose a tile grid from the source image dimensions, respecting {@code maxTiles}
     * 2. Crop each tile from the original aspect-ratio image
     * 3. Resize each tile to fit within {@code maxSize x maxSize}, then pad to square
     * 4. Add a global image frame using the same resize-to-fit + pad behavior
     *
     * @param image The input image
     * @param maxSize The maximum tile size (e.g. 512 for SmolDocling)
     * @return SplitImageResult containing the frames and metadata
     */
    public static SplitImageResult splitImageForVLM(BufferedImage image, int maxSize) {
        return splitImageForVLM(image, maxSize, -1);
    }

    /**
     * Split an image into tiles for VLM processing with a maximum tile limit.
     *
     * @param image The input image
     * @param maxSize The maximum tile size (e.g. 512 for SmolDocling)
     * @param maxTiles Maximum number of content tiles (-1 for no limit)
     * @return SplitImageResult containing the frames and metadata
     */
    public static SplitImageResult splitImageForVLM(BufferedImage image, int maxSize, int maxTiles) {
        // Step 1: resize_for_vision_encoder — resize so dimensions are multiples of maxSize.
        // This matches the HuggingFace Idefics3 preprocessing that the model was trained with.
        int width = image.getWidth();
        int height = image.getHeight();
        if (height > maxSize || width > maxSize) {
            double aspectRatio = (double) width / height;
            int newWidth, newHeight;
            if (width >= height) {
                newWidth = (int) Math.ceil((double) width / maxSize) * maxSize;
                newHeight = (int) (newWidth / aspectRatio);
                newHeight = (int) Math.ceil((double) newHeight / maxSize) * maxSize;
            } else {
                newHeight = (int) Math.ceil((double) height / maxSize) * maxSize;
                newWidth = (int) (newHeight * aspectRatio);
                newWidth = (int) Math.ceil((double) newWidth / maxSize) * maxSize;
            }
            log.info("resize_for_vision_encoder: {}x{} -> {}x{} (multiples of {})",
                    width, height, newWidth, newHeight, maxSize);
            image = resizeImage(image, newWidth, newHeight);
            width = newWidth;
            height = newHeight;
        }

        log.info("Splitting image {}x{} into {}x{} tiles (maxTiles={})", width, height, maxSize, maxSize,
                maxTiles > 0 ? maxTiles : "unlimited");

        List<BufferedImage> frames = new ArrayList<>();
        List<ContentRegion> contentRegions = new ArrayList<>();
        int numSplitsH = 0;
        int numSplitsW = 0;

        if (maxTiles == 1) {
            log.info("maxTiles=1: skipping tiling, using single global image only");
        } else if (height > maxSize || width > maxSize) {
            numSplitsH = (int) Math.ceil((double) height / maxSize);
            numSplitsW = (int) Math.ceil((double) width / maxSize);

            if (maxTiles > 0) {
                int totalTiles = numSplitsH * numSplitsW;
                if (totalTiles > maxTiles) {
                    double imageAspect = (double) height / width;
                    int bestH = 1, bestW = 1;
                    int bestCount = 1;
                    double bestAspectMatch = Double.MAX_VALUE;
                    for (int h = 1; h <= Math.min(numSplitsH, maxTiles); h++) {
                        int maxW = maxTiles / h;
                        for (int w = 1; w <= Math.min(numSplitsW, maxW); w++) {
                            double gridAspect = (double) h / w;
                            double aspectMatch = Math.abs(gridAspect - imageAspect);
                            int count = h * w;
                            if (count > bestCount || (count == bestCount && aspectMatch < bestAspectMatch)) {
                                bestH = h;
                                bestW = w;
                                bestCount = count;
                                bestAspectMatch = aspectMatch;
                            }
                        }
                    }
                    numSplitsH = bestH;
                    numSplitsW = bestW;
                }
            }

            int optimalHeight = (int) Math.ceil((double) height / numSplitsH);
            int optimalWidth = (int) Math.ceil((double) width / numSplitsW);

            log.info("Splitting into {}x{} grid ({} tiles), optimal tile size: {}x{}",
                    numSplitsH, numSplitsW, numSplitsH * numSplitsW, optimalHeight, optimalWidth);

            for (int r = 0; r < numSplitsH; r++) {
                for (int c = 0; c < numSplitsW; c++) {
                    int startX = c * optimalWidth;
                    int startY = r * optimalHeight;
                    int endX = Math.min(startX + optimalWidth, width);
                    int endY = Math.min(startY + optimalHeight, height);

                    int tileWidth = endX - startX;
                    int tileHeight = endY - startY;

                    BufferedImage tile = image.getSubimage(startX, startY, tileWidth, tileHeight);
                    // Resize tile to exactly maxSize x maxSize (matching HF Idefics3 preprocessing)
                    if (tileWidth != maxSize || tileHeight != maxSize) {
                        tile = resizeImage(tile, maxSize, maxSize);
                    }

                    frames.add(tile);
                    contentRegions.add(new ContentRegion(maxSize, maxSize));
                    log.debug("  Tile [{},{}]: crop ({},{}) to ({},{}), tile {}x{} -> {}x{}",
                            r, c, startX, startY, endX, endY, tileWidth, tileHeight, maxSize, maxSize);
                }
            }
        }

        // Global image resized to maxSize x maxSize (squished, matching HF Idefics3)
        BufferedImage globalImage = resizeImage(image, maxSize, maxSize);
        frames.add(globalImage);
        contentRegions.add(new ContentRegion(maxSize, maxSize));
        log.debug("  Added global resized image ({}x{}, squished)", maxSize, maxSize);

        log.info("Total frames: {} ({} tiles + 1 global)", frames.size(), frames.size() - 1);

        return new SplitImageResult(frames, contentRegions, numSplitsH, numSplitsW);
    }

    /**
     * Split an image into tiles using parallel tile extraction and resizing.
     * Each tile crop + resize runs concurrently on a thread pool.
     *
     * @param image The input image
     * @param maxSize The maximum tile size (e.g. 512 for SmolDocling)
     * @param maxTiles Maximum number of content tiles (-1 for no limit)
     * @param numThreads Number of threads for parallel tile extraction
     * @return SplitImageResult containing the frames and metadata
     */
    public static SplitImageResult splitImageForVLMParallel(BufferedImage image, int maxSize, int maxTiles, int numThreads) {
        // Step 1: resize_for_vision_encoder — resize so dimensions are multiples of maxSize.
        int width = image.getWidth();
        int height = image.getHeight();
        if (height > maxSize || width > maxSize) {
            double aspectRatio = (double) width / height;
            int newWidth, newHeight;
            if (width >= height) {
                newWidth = (int) Math.ceil((double) width / maxSize) * maxSize;
                newHeight = (int) (newWidth / aspectRatio);
                newHeight = (int) Math.ceil((double) newHeight / maxSize) * maxSize;
            } else {
                newHeight = (int) Math.ceil((double) height / maxSize) * maxSize;
                newWidth = (int) (newHeight * aspectRatio);
                newWidth = (int) Math.ceil((double) newWidth / maxSize) * maxSize;
            }
            image = resizeImage(image, newWidth, newHeight);
            width = newWidth;
            height = newHeight;
        }

        int numSplitsH = 0;
        int numSplitsW = 0;
        if (maxTiles != 1 && (height > maxSize || width > maxSize)) {
            numSplitsH = (int) Math.ceil((double) height / maxSize);
            numSplitsW = (int) Math.ceil((double) width / maxSize);
            if (maxTiles > 0 && numSplitsH * numSplitsW > maxTiles) {
                double imageAspect = (double) height / width;
                int bestH = 1, bestW = 1;
                int bestCount = 1;
                double bestAspectMatch = Double.MAX_VALUE;
                for (int h = 1; h <= Math.min(numSplitsH, maxTiles); h++) {
                    int maxW = maxTiles / h;
                    for (int w = 1; w <= Math.min(numSplitsW, maxW); w++) {
                        double gridAspect = (double) h / w;
                        double aspectMatch = Math.abs(gridAspect - imageAspect);
                        int count = h * w;
                        if (count > bestCount || (count == bestCount && aspectMatch < bestAspectMatch)) {
                            bestH = h;
                            bestW = w;
                            bestCount = count;
                            bestAspectMatch = aspectMatch;
                        }
                    }
                }
                numSplitsH = bestH;
                numSplitsW = bestW;
            }
        }

        int totalTiles = numSplitsH * numSplitsW;
        int totalFrames = totalTiles + 1; // tiles + global

        List<BufferedImage> frames = new ArrayList<>(totalFrames);
        List<ContentRegion> contentRegions = new ArrayList<>(totalFrames);

        if (totalTiles == 0) {
            BufferedImage globalImage = resizeImage(image, maxSize, maxSize);
            frames.add(globalImage);
            contentRegions.add(new ContentRegion(maxSize, maxSize));
            return new SplitImageResult(frames, contentRegions, 0, 0);
        }

        int optimalHeight = (int) Math.ceil((double) height / numSplitsH);
        int optimalWidth = (int) Math.ceil((double) width / numSplitsW);

        // Pre-size lists
        for (int i = 0; i < totalFrames; i++) {
            frames.add(null);
            contentRegions.add(null);
        }

        int effectiveThreads = Math.min(numThreads, totalFrames);
        final BufferedImage sourceImage = image;
        final int srcWidth = width;
        final int srcHeight = height;
        final int nH = numSplitsH;
        final int nW = numSplitsW;

        if (effectiveThreads <= 1) {
            // Sequential fallback
            for (int r = 0; r < nH; r++) {
                for (int c = 0; c < nW; c++) {
                    int idx = r * nW + c;
                    int startX = c * optimalWidth;
                    int startY = r * optimalHeight;
                    int endX = Math.min(startX + optimalWidth, srcWidth);
                    int endY = Math.min(startY + optimalHeight, srcHeight);
                    BufferedImage tile = sourceImage.getSubimage(startX, startY, endX - startX, endY - startY);
                    int tw = endX - startX, th = endY - startY;
                    if (tw != maxSize || th != maxSize) {
                        tile = resizeImage(tile, maxSize, maxSize);
                    }
                    frames.set(idx, tile);
                    contentRegions.set(idx, new ContentRegion(maxSize, maxSize));
                }
            }
            BufferedImage globalImage = resizeImage(sourceImage, maxSize, maxSize);
            frames.set(totalTiles, globalImage);
            contentRegions.set(totalTiles, new ContentRegion(maxSize, maxSize));
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(effectiveThreads, r -> {
                Thread t = new Thread(r, "ImageTiler");
                t.setDaemon(true);
                return t;
            });
            try {
                List<Future<?>> futures = new ArrayList<>(totalFrames);
                for (int r = 0; r < nH; r++) {
                    for (int c = 0; c < nW; c++) {
                        final int row = r, col = c;
                        final int idx = r * nW + c;
                        futures.add(pool.submit(() -> {
                            int startX = col * optimalWidth;
                            int startY = row * optimalHeight;
                            int endX = Math.min(startX + optimalWidth, srcWidth);
                            int endY = Math.min(startY + optimalHeight, srcHeight);
                            BufferedImage tile = sourceImage.getSubimage(startX, startY, endX - startX, endY - startY);
                            int tw = endX - startX, th = endY - startY;
                            if (tw != maxSize || th != maxSize) {
                                tile = resizeImage(tile, maxSize, maxSize);
                            }
                            frames.set(idx, tile);
                            contentRegions.set(idx, new ContentRegion(maxSize, maxSize));
                        }));
                    }
                }
                // Global image as last frame (squished)
                futures.add(pool.submit(() -> {
                    BufferedImage globalImage = resizeImage(sourceImage, maxSize, maxSize);
                    frames.set(totalTiles, globalImage);
                    contentRegions.set(totalTiles, new ContentRegion(maxSize, maxSize));
                }));
                for (Future<?> future : futures) {
                    future.get();
                }
            } catch (Exception e) {
                throw new RuntimeException("Parallel tile extraction failed", e);
            } finally {
                pool.shutdown();
            }
        }

        log.info("Parallel tiling: {}x{} grid ({} tiles + 1 global), {} threads",
                nH, nW, totalTiles, effectiveThreads);

        return new SplitImageResult(frames, contentRegions, numSplitsH, numSplitsW);
    }

    private static int[] chooseGrid(int height, int width, int maxSize, int maxTiles) {
        int numSplitsH = 0;
        int numSplitsW = 0;

        if (maxTiles == 1 || (height <= maxSize && width <= maxSize)) {
            return new int[]{0, 0};
        }

        numSplitsH = (int) Math.ceil((double) height / maxSize);
        numSplitsW = (int) Math.ceil((double) width / maxSize);

        if (maxTiles > 0 && numSplitsH * numSplitsW > maxTiles) {
            double imageAspect = (double) height / width;
            int bestH = 1;
            int bestW = 1;
            int bestCount = 1;
            double bestAspectMatch = Double.MAX_VALUE;

            for (int h = 1; h <= Math.min(numSplitsH, maxTiles); h++) {
                int maxW = maxTiles / h;
                for (int w = 1; w <= Math.min(numSplitsW, maxW); w++) {
                    int count = h * w;
                    if (count <= maxTiles) {
                        double gridAspect = (double) h / w;
                        double aspectMatch = Math.abs(Math.log(gridAspect) - Math.log(imageAspect));
                        if (count > bestCount || (count == bestCount && aspectMatch < bestAspectMatch)) {
                            bestH = h;
                            bestW = w;
                            bestCount = count;
                            bestAspectMatch = aspectMatch;
                        }
                    }
                }
            }

            log.info("Reduced grid from {}x{} to {}x{} ({} tiles) to fit maxTiles={}, imageAspect={}",
                    numSplitsH, numSplitsW, bestH, bestW, bestH * bestW, maxTiles, imageAspect);
            numSplitsH = bestH;
            numSplitsW = bestW;
        }

        int maxGrid = 6;
        if (numSplitsH > maxGrid || numSplitsW > maxGrid) {
            double scale = Math.min((double) maxGrid / numSplitsH, (double) maxGrid / numSplitsW);
            int newH = Math.max(1, (int) Math.floor(numSplitsH * scale));
            int newW = Math.max(1, (int) Math.floor(numSplitsW * scale));
            numSplitsH = Math.min(maxGrid, newH);
            numSplitsW = Math.min(maxGrid, newW);
            log.info("Reducing grid {}x{} to {}x{} to fit row/col token limits",
                    (int) Math.ceil((double) height / maxSize),
                    (int) Math.ceil((double) width / maxSize),
                    numSplitsH,
                    numSplitsW);
        }

        return new int[]{numSplitsH, numSplitsW};
    }

    private static PreparedFrame prepareFrame(BufferedImage source, int maxSize) {
        ResizeResult resized = resizeToFit(source, maxSize, maxSize);
        BufferedImage padded = padToSize(resized.image, maxSize, maxSize);
        return new PreparedFrame(padded, new ContentRegion(resized.width, resized.height));
    }

    private static class PreparedFrame {
        private final BufferedImage image;
        private final ContentRegion contentRegion;

        private PreparedFrame(BufferedImage image, ContentRegion contentRegion) {
            this.image = image;
            this.contentRegion = contentRegion;
        }
    }

    /**
     * Resize image so that the longest edge matches the target length.
     */
    public static BufferedImage resizeLongestEdge(BufferedImage image, int longestEdge) {
        int width = image.getWidth();
        int height = image.getHeight();
        int maxDim = Math.max(width, height);
        if (maxDim == longestEdge) {
            return image;
        }
        double scale = (double) longestEdge / (double) maxDim;
        int newW = Math.max(1, (int) Math.round(width * scale));
        int newH = Math.max(1, (int) Math.round(height * scale));
        return resizeImage(image, newW, newH);
    }

    /**
     * Resize an image to fit within the target size while preserving aspect ratio.
     */
    public static ResizeResult resizeToFit(BufferedImage image, int targetWidth, int targetHeight) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width <= targetWidth && height <= targetHeight) {
            return new ResizeResult(image, width, height);
        }
        double scale = Math.min((double) targetWidth / (double) width, (double) targetHeight / (double) height);
        int newW = Math.max(1, (int) Math.round(width * scale));
        int newH = Math.max(1, (int) Math.round(height * scale));
        return new ResizeResult(resizeImage(image, newW, newH), newW, newH);
    }

    /**
     * Resize an image to the specified dimensions using bicubic interpolation.
     */
    public static BufferedImage resizeImage(BufferedImage original, int targetWidth, int targetHeight) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();

        // Apply mild sharpening to preserve character edge clarity for OCR
        String sharpen = System.getProperty("nd4j.vlm.image.sharpen", "false");
        if ("true".equalsIgnoreCase(sharpen)) {
            float[] sharpenKernel = {
                0, -0.5f, 0,
                -0.5f, 3.0f, -0.5f,
                0, -0.5f, 0
            };
            Kernel kernel = new Kernel(3, 3, sharpenKernel);
            ConvolveOp sharpenOp = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
            resized = sharpenOp.filter(resized, null);
        }

        return resized;
    }

    /**
     * Pad an image to the target size (top-left aligned, white padding).
     */
    public static BufferedImage padToSize(BufferedImage image, int targetWidth, int targetHeight) {
        if (image.getWidth() == targetWidth && image.getHeight() == targetHeight) {
            return image;
        }
        BufferedImage padded = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = padded.createGraphics();
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, targetWidth, targetHeight);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return padded;
    }

    /**
     * Create a pixel attention mask for a padded frame.
     * Mask is 1 (true) where content exists, 0 (false) for padding.
     *
     * @param contentWidth actual content width
     * @param contentHeight actual content height
     * @param targetSize padded frame size
     * @return INDArray of shape [1, targetSize, targetSize] with BOOL dtype
     */
    public static INDArray createPixelAttentionMask(int contentWidth, int contentHeight, int targetSize) {
        if (contentWidth >= targetSize && contentHeight >= targetSize) {
            return Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        }
        INDArray mask = Nd4j.zeros(DataType.BOOL, 1, targetSize, targetSize);
        if (contentWidth > 0 && contentHeight > 0) {
            mask.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.interval(0, contentHeight),
                    NDArrayIndex.interval(0, contentWidth)
            ).assign(1);
        }
        return mask;
    }
}
