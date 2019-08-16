package org.deeplearning4j.rl4j.util;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.opencv.imgproc.Imgproc;

import static org.bytedeco.ffmpeg.global.avcodec.*;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC;

/**
 * VideoRecorder is used to create a video from a sequence of individual frames. If using 3 channels
 * images, it expects B-G-R order. A RGB order can be used by calling isRGBOrder(true).<br>
 * Example:<br>
 * <pre>
 * {@code
 *        VideoRecorder recorder = VideoRecorder.builder(160, 100)
 *             .numChannels(3)
 *             .isRGBOrder(true)
 *             .build();
 *         recorder.startRecording("myVideo.mp4");
 *         while(...) {
 *             byte[] data = new byte[160*100*3];
 *             // Todo: Fill data
 *             VideoRecorder.VideoFrame frame = recorder.createFrame(data);
 *             // Todo: Apply cropping or resizing to frame
 *             recorder.record(frame);
 *         }
 *         recorder.stopRecording();
 * }
 * </pre>
 */
@Slf4j
public class VideoRecorder implements AutoCloseable {

    private final int height;
    private final int width;
    private final int imageType;
    private final OpenCVFrameConverter openCVFrameConverter = new OpenCVFrameConverter.ToMat();
    private final int codec;
    private final double framerate;
    private final int videoQuality;
    private final boolean isRGBOrder;

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
        imageType = CV_8UC(builder.numChannels);
        codec = builder.codec;
        framerate = builder.frameRate;
        videoQuality = builder.videoQuality;
        isRGBOrder = builder.isRGBOrder;
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
     * @param frame the VideoFrame to add to the video
     * @throws Exception
     */
    public void record(VideoFrame frame) throws Exception {
        Size size = frame.getMat().size();
        if(size.height() != height || size.width() != width) {
            throw new IllegalArgumentException(String.format("Wrong frame size. Got (%dh x %dw) expected (%dh x %dw)", size.height(), size.width(), height, width));
        }
        Frame cvFrame = openCVFrameConverter.convert(frame.getMat());
        fmpegFrameRecorder.record(cvFrame);
    }

    /**
     * Create a VideoFrame from a byte array.
     * @param data A byte array. Expect the index to be of the form [(Y*Width + X) * NumChannels + channel]
     * @return An instance of VideoFrame
     */
    public VideoFrame createFrame(byte[] data) {
        return createFrame(new BytePointer(data));
    }

    /**
     * Create a VideoFrame from a byte array with different height and width than the video
     * the frame will need to be cropped or resized before being added to the video)
     *
     * @param data A byte array Expect the index to be of the form [(Y*customWidth + X) * NumChannels + channel]
     * @param customHeight The actual height of the data
     * @param customWidth The actual width of the data
     * @return A VideoFrame instance
     */
    public VideoFrame createFrame(byte[] data, int customHeight, int customWidth) {
        return createFrame(new BytePointer(data), customHeight, customWidth);
    }

    /**
     * Create a VideoFrame from a Pointer (to use for example with a INDarray).
     * @param data A Pointer (for example myINDArray.data().pointer())
     * @return An instance of VideoFrame
     */
    public VideoFrame createFrame(Pointer data) {
        return new VideoFrame(height, width, imageType, isRGBOrder, data);
    }

    /**
     *  Create a VideoFrame from a Pointer with different height and width than the video
     * the frame will need to be cropped or resized before being added to the video)
     * @param data
     * @param customHeight The actual height of the data
     * @param customWidth The actual width of the data
     * @return A VideoFrame instance
     */
    public VideoFrame createFrame(Pointer data, int customHeight, int customWidth) {
        return new VideoFrame(customHeight, customWidth, imageType, isRGBOrder, data);
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
     * An individual frame for the video
     */
    public static class VideoFrame {

        private final int height;
        private final int width;
        private final int imageType;
        @Getter
        private Mat mat;

        private VideoFrame(int height, int width, int imageType, boolean isRGBOrder, Pointer data) {
            this.height = height;
            this.width = width;
            this.imageType = imageType;
            if(isRGBOrder) {
                Mat src = new Mat(height, width, imageType, data);
                mat = new Mat(height, width, imageType);
                opencv_imgproc.cvtColor(src, mat, Imgproc.COLOR_RGB2BGR);
            }
            else {
                mat = new Mat(height, width, imageType, data);
            }
        }

        /**
         * Crop the video to a specified size
         * @param newHeight The new height of the frame
         * @param newWidth The new width of the frame
         * @param heightOffset The starting height offset in the uncropped frame
         * @param widthOffset The starting weight offset in the uncropped frame
         */
        public void crop(int newHeight, int newWidth, int heightOffset, int widthOffset) {
            mat = mat.apply(new Rect(widthOffset, heightOffset, newWidth, newHeight));
        }

        /**
         * Resize the frame to a specified size
         * @param newHeight The new height of the frame
         * @param newWidth The new width of the frame
         */
        public void resize(int newHeight, int newWidth) {
            mat = new Mat(newHeight, newWidth, imageType);
        }
    }

    /**
     * A builder class for the VideoRecorder
     */
    public static class Builder {
        private final int height;
        private final int width;
        private int numChannels = 3;
        private boolean isRGBOrder = false;
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
         * Specify the number of channels. Default is 3 (B-G-R)
         * @param numChannels
         */
        public Builder numChannels(int numChannels) {
            this.numChannels = numChannels;
            return this;
        }

        /**
         * Tell the VideoRecorder that data will be in the R-G-B order (isRGBOrder(true))
         * @param isRGBOrder
         */
        public Builder isRGBOrder(boolean isRGBOrder) {
            this.isRGBOrder = isRGBOrder;
            return this;
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
