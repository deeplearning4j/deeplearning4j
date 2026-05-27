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

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Extracts frames from video files for VLM processing.
 *
 * <p>Uses JavaCV/FFmpeg for video decoding. Supports MP4, AVI, MKV, WebM, and other
 * FFmpeg-supported formats. JavaCV dependencies are optional — if not on the classpath,
 * methods throw {@link UnsupportedOperationException} with a helpful message.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VideoFrameExtractor extractor = VideoFrameExtractor.builder().build();
 *
 * // Extract up to 32 uniformly-spaced frames
 * List<BufferedImage> frames = extractor.extractFrames(new File("video.mp4"), 32);
 *
 * // Extract at a specific FPS
 * List<BufferedImage> frames = extractor.extractAtFPS(new File("video.mp4"), 1.0);
 * }</pre>
 *
 * @see VideoFrameSampler
 * @see VideoPreprocessor
 */
@Slf4j
@Builder
public class VideoFrameExtractor implements AutoCloseable {

    private static final boolean JAVACV_AVAILABLE;

    static {
        boolean available = false;
        try {
            Class.forName("org.bytedeco.javacv.FFmpegFrameGrabber");
            available = true;
        } catch (ClassNotFoundException e) {
            // JavaCV not on classpath
        }
        JAVACV_AVAILABLE = available;
    }

    @Getter
    @Builder.Default
    private int maxFrames = 64;

    @Getter
    @Builder.Default
    private double targetFPS = -1;

    private static void checkJavaCVAvailable() {
        if (!JAVACV_AVAILABLE) {
            throw new UnsupportedOperationException(
                    "JavaCV is required for video frame extraction but is not on the classpath. " +
                    "Add org.bytedeco:javacv and org.bytedeco:ffmpeg to your dependencies.");
        }
    }

    /**
     * Extract uniformly-spaced frames from a video file.
     *
     * @param videoFile the video file to extract frames from
     * @param maxFrames maximum number of frames to extract
     * @return list of extracted frames as BufferedImages
     * @throws IOException if the video cannot be read
     */
    public List<BufferedImage> extractFrames(File videoFile, int maxFrames) throws IOException {
        checkJavaCVAvailable();
        if (!videoFile.exists()) {
            throw new IOException("Video file not found: " + videoFile.getAbsolutePath());
        }

        return JavaCVFrameExtractor.extractUniform(videoFile, maxFrames);
    }

    /**
     * Extract frames from a video file at the configured max frames count.
     *
     * @param videoFile the video file
     * @return list of extracted frames
     * @throws IOException if the video cannot be read
     */
    public List<BufferedImage> extractFrames(File videoFile) throws IOException {
        return extractFrames(videoFile, maxFrames);
    }

    /**
     * Extract frames from a video at a specific frame rate.
     *
     * @param videoFile the video file
     * @param fps target frames per second to extract
     * @return list of extracted frames
     * @throws IOException if the video cannot be read
     */
    public List<BufferedImage> extractAtFPS(File videoFile, double fps) throws IOException {
        return extractAtFPS(videoFile, fps, Integer.MAX_VALUE);
    }

    /**
     * Extract frames from a video at a specific frame rate with a maximum count.
     *
     * @param videoFile the video file
     * @param fps target frames per second to extract
     * @param maxFrames maximum number of frames to extract
     * @return list of extracted frames
     * @throws IOException if the video cannot be read
     */
    public List<BufferedImage> extractAtFPS(File videoFile, double fps, int maxFrames) throws IOException {
        checkJavaCVAvailable();
        if (!videoFile.exists()) {
            throw new IOException("Video file not found: " + videoFile.getAbsolutePath());
        }

        return JavaCVFrameExtractor.extractAtFPS(videoFile, fps, maxFrames);
    }

    /**
     * Extract only keyframes (I-frames) from a video file.
     *
     * @param videoFile the video file
     * @param maxFrames maximum number of keyframes to extract
     * @return list of extracted keyframes
     * @throws IOException if the video cannot be read
     */
    public List<BufferedImage> extractKeyframes(File videoFile, int maxFrames) throws IOException {
        checkJavaCVAvailable();
        if (!videoFile.exists()) {
            throw new IOException("Video file not found: " + videoFile.getAbsolutePath());
        }

        return JavaCVFrameExtractor.extractKeyframes(videoFile, maxFrames);
    }

    /**
     * Get video metadata without extracting frames.
     *
     * @param videoFile the video file
     * @return video metadata
     * @throws IOException if the video cannot be read
     */
    public VideoMetadata getMetadata(File videoFile) throws IOException {
        checkJavaCVAvailable();
        if (!videoFile.exists()) {
            throw new IOException("Video file not found: " + videoFile.getAbsolutePath());
        }

        return JavaCVFrameExtractor.getMetadata(videoFile);
    }

    /**
     * Check if JavaCV is available on the classpath.
     *
     * @return true if JavaCV can be used for video extraction
     */
    public static boolean isJavaCVAvailable() {
        return JAVACV_AVAILABLE;
    }

    @Override
    public void close() {
        // No persistent state to clean up
    }

    /**
     * Video file metadata.
     */
    @Getter
    @Builder
    public static class VideoMetadata {
        private final int width;
        private final int height;
        private final double fps;
        private final long totalFrames;
        private final double durationSeconds;
        private final String codecName;
    }

    /**
     * Inner class that isolates all JavaCV references so they only load if JavaCV is present.
     */
    static class JavaCVFrameExtractor {

        static List<BufferedImage> extractUniform(File videoFile, int maxFrames) throws IOException {
            try {
                org.bytedeco.javacv.FFmpegFrameGrabber grabber = new org.bytedeco.javacv.FFmpegFrameGrabber(videoFile);
                grabber.start();

                int totalFrames = grabber.getLengthInFrames();
                double fps = grabber.getFrameRate();
                log.info("Video: {}x{}, {:.1f} fps, {} frames, {:.1f}s",
                        grabber.getImageWidth(), grabber.getImageHeight(),
                        fps, totalFrames, totalFrames / Math.max(fps, 1.0));

                int framesToExtract = Math.min(maxFrames, totalFrames);
                double step = totalFrames > framesToExtract ? (double) totalFrames / framesToExtract : 1.0;

                List<BufferedImage> frames = new ArrayList<>(framesToExtract);
                org.bytedeco.javacv.Java2DFrameConverter converter = new org.bytedeco.javacv.Java2DFrameConverter();

                for (int i = 0; i < framesToExtract && frames.size() < maxFrames; i++) {
                    int frameIndex = (int) Math.round(i * step);
                    grabber.setFrameNumber(Math.min(frameIndex, totalFrames - 1));

                    org.bytedeco.javacv.Frame frame = grabber.grabImage();
                    if (frame != null && frame.image != null) {
                        BufferedImage img = converter.convert(frame);
                        if (img != null) {
                            // Deep copy since the frame buffer is reused
                            BufferedImage copy = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
                            copy.getGraphics().drawImage(img, 0, 0, null);
                            frames.add(copy);
                        }
                    }
                }

                grabber.stop();
                grabber.close();

                log.info("Extracted {} frames from {}", frames.size(), videoFile.getName());
                return frames;
            } catch (Exception e) {
                throw new IOException("Failed to extract frames from video: " + videoFile, e);
            }
        }

        static List<BufferedImage> extractAtFPS(File videoFile, double targetFPS, int maxFrames) throws IOException {
            try {
                org.bytedeco.javacv.FFmpegFrameGrabber grabber = new org.bytedeco.javacv.FFmpegFrameGrabber(videoFile);
                grabber.start();

                double videoFPS = grabber.getFrameRate();
                int totalFrames = grabber.getLengthInFrames();
                double durationSec = totalFrames / Math.max(videoFPS, 1.0);
                int expectedFrames = Math.min((int) Math.ceil(durationSec * targetFPS), maxFrames);

                List<BufferedImage> frames = new ArrayList<>(expectedFrames);
                org.bytedeco.javacv.Java2DFrameConverter converter = new org.bytedeco.javacv.Java2DFrameConverter();

                double frameInterval = videoFPS / targetFPS;
                double nextFramePos = 0;

                for (int i = 0; i < totalFrames && frames.size() < maxFrames; i++) {
                    org.bytedeco.javacv.Frame frame = grabber.grabImage();
                    if (frame == null) break;

                    if (i >= nextFramePos) {
                        if (frame.image != null) {
                            BufferedImage img = converter.convert(frame);
                            if (img != null) {
                                BufferedImage copy = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
                                copy.getGraphics().drawImage(img, 0, 0, null);
                                frames.add(copy);
                            }
                        }
                        nextFramePos += frameInterval;
                    }
                }

                grabber.stop();
                grabber.close();

                log.info("Extracted {} frames at {:.1f} FPS from {} ({:.1f}s video at {:.1f} FPS)",
                        frames.size(), targetFPS, videoFile.getName(), durationSec, videoFPS);
                return frames;
            } catch (Exception e) {
                throw new IOException("Failed to extract frames at FPS from video: " + videoFile, e);
            }
        }

        static List<BufferedImage> extractKeyframes(File videoFile, int maxFrames) throws IOException {
            try {
                org.bytedeco.javacv.FFmpegFrameGrabber grabber = new org.bytedeco.javacv.FFmpegFrameGrabber(videoFile);
                grabber.start();

                List<BufferedImage> frames = new ArrayList<>();
                org.bytedeco.javacv.Java2DFrameConverter converter = new org.bytedeco.javacv.Java2DFrameConverter();

                org.bytedeco.javacv.Frame frame;
                while ((frame = grabber.grabKeyFrame()) != null && frames.size() < maxFrames) {
                    if (frame.image != null) {
                        BufferedImage img = converter.convert(frame);
                        if (img != null) {
                            BufferedImage copy = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
                            copy.getGraphics().drawImage(img, 0, 0, null);
                            frames.add(copy);
                        }
                    }
                }

                grabber.stop();
                grabber.close();

                log.info("Extracted {} keyframes from {}", frames.size(), videoFile.getName());
                return frames;
            } catch (Exception e) {
                throw new IOException("Failed to extract keyframes from video: " + videoFile, e);
            }
        }

        static VideoMetadata getMetadata(File videoFile) throws IOException {
            try {
                org.bytedeco.javacv.FFmpegFrameGrabber grabber = new org.bytedeco.javacv.FFmpegFrameGrabber(videoFile);
                grabber.start();

                VideoMetadata metadata = VideoMetadata.builder()
                        .width(grabber.getImageWidth())
                        .height(grabber.getImageHeight())
                        .fps(grabber.getFrameRate())
                        .totalFrames(grabber.getLengthInFrames())
                        .durationSeconds(grabber.getLengthInFrames() / Math.max(grabber.getFrameRate(), 1.0))
                        .codecName(grabber.getVideoCodecName())
                        .build();

                grabber.stop();
                grabber.close();

                return metadata;
            } catch (Exception e) {
                throw new IOException("Failed to read video metadata: " + videoFile, e);
            }
        }
    }
}
