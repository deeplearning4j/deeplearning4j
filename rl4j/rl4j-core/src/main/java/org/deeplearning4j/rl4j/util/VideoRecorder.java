/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_H264;

/**
 * VideoRecorder is used to create a video from a sequence of INDArray frames. INDArrays are assumed to be in CHW format where C=3 and pixels are RGB encoded<br>
 * Example:<br>
 * <pre>
 * {@code
 *        VideoRecorder recorder = VideoRecorder.builder(160, 100)
 *             .numChannels(3)
 *             .isRGBOrder(true)
 *             .build();
 *         recorder.startRecording("myVideo.mp4");
 *         while(...) {
 *             INDArray chwData = Nd4j.create()
 *             recorder.record(chwData);
 *         }
 *         recorder.stopRecording();
 * }
 * </pre>
 *
 * @author Alexandre Boulanger
 */
@Slf4j
public class VideoRecorder implements AutoCloseable {

    private final NativeImageLoader nativeImageLoader = new NativeImageLoader();

    private final int height;
    private final int width;
    private final int codec;
    private final double framerate;
    private final int videoQuality;

    private FFmpegFrameRecorder fmpegFrameRecorder = null;

    /**
     * @return True if the instance is recording a video
     */
    public boolean isRecording() {
        return fmpegFrameRecorder != null;
    }

    private VideoRecorder(Builder builder) {
        this.height = builder.height;
        this.width = builder.width;
        codec = builder.codec;
        framerate = builder.frameRate;
        videoQuality = builder.videoQuality;
    }

    /**
     * Initiate the recording of a video
     * @param filename Name of the video file to create
     * @throws Exception
     */
    public void startRecording(String filename) throws Exception {
        stopRecording();

        fmpegFrameRecorder = new FFmpegFrameRecorder(filename, width, height);
        fmpegFrameRecorder.setVideoCodec(codec);
        fmpegFrameRecorder.setFrameRate(framerate);
        fmpegFrameRecorder.setVideoQuality(videoQuality);
        fmpegFrameRecorder.start();
    }

    /**
     * Terminates the recording of the video
     * @throws Exception
     */
    public void stopRecording() throws Exception {
        if (fmpegFrameRecorder != null) {
            fmpegFrameRecorder.stop();
            fmpegFrameRecorder.release();
        }
        fmpegFrameRecorder = null;
    }

    /**
     * Add a frame to the video
     * @param imageArray the INDArray that contains the data to be recorded, the data must be in CHW format
     * @throws Exception
     */
    public void record(INDArray imageArray) throws Exception {
        fmpegFrameRecorder.record(nativeImageLoader.asFrame(imageArray, Frame.DEPTH_UBYTE));
    }

    /**
     * Terminate the recording and close the video file
     * @throws Exception
     */
    public void close() throws Exception {
        stopRecording();
    }

    /**
     *
     * @param height The height of the video
     * @param width Thw width of the video
     * @return A VideoRecorder builder
     */
    public static Builder builder(int height, int width) {
        return new Builder(height, width);
    }

    /**
     * A builder class for the VideoRecorder
     */
    public static class Builder {
        private final int height;
        private final int width;
        private int codec = AV_CODEC_ID_H264;
        private double frameRate = 30.0;
        private int videoQuality = 30;

        /**
         * @param height The height of the video
         * @param width The width of the video
         */
        public Builder(int height, int width) {
            this.height = height;
            this.width = width;
        }

        /**
         * The codec to use for the video. Default is AV_CODEC_ID_H264
         * @param codec Code (see {@link org.bytedeco.ffmpeg.global.avcodec codec codes})
         */
        public Builder codec(int codec) {
            this.codec = codec;
            return this;
        }

        /**
         * The frame rate of the video. Default is 30.0
         * @param frameRate The frame rate
         * @return
         */
        public Builder frameRate(double frameRate) {
            this.frameRate = frameRate;
            return this;
        }

        /**
         * The video quality. Default is 30
         * @param videoQuality
         * @return
         */
        public Builder videoQuality(int videoQuality) {
            this.videoQuality = videoQuality;
            return this;
        }

        /**
         * Build an instance of the configured VideoRecorder
         * @return A VideoRecorder instance
         */
        public VideoRecorder build() {
            return new VideoRecorder(this);
        }
    }
}
