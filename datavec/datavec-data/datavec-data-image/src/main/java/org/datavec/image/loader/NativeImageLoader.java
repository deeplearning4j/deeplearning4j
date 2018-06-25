/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */
package org.datavec.image.loader;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.Image;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.*;
import java.nio.ByteOrder;

import static org.bytedeco.javacpp.lept.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Uses JavaCV to load images. Allowed formats: bmp, gif, jpg, jpeg, jp2, pbm, pgm, ppm, pnm, png, tif, tiff, exr, webp
 *
 * @author saudet
 */
public class NativeImageLoader extends BaseImageLoader {
    private static final int MIN_BUFFER_STEP_SIZE = 64 * 1024;
    private byte[] buffer = null;
    private Mat bufferMat = null;

    public static final String[] ALLOWED_FORMATS = {"bmp", "gif", "jpg", "jpeg", "jp2", "pbm", "pgm", "ppm", "pnm",
                    "png", "tif", "tiff", "exr", "webp", "BMP", "GIF", "JPG", "JPEG", "JP2", "PBM", "PGM", "PPM", "PNM",
                    "PNG", "TIF", "TIFF", "EXR", "WEBP"};

    protected OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

    boolean direct = !Loader.getPlatform().startsWith("android");

    /**
     * Loads images with no scaling or conversion.
     */
    public NativeImageLoader() {}

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
    
     */
    public NativeImageLoader(long height, long width) {
        this.height = height;
        this.width = width;
    }


    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     */
    public NativeImageLoader(long height, long width, long channels) {
        this.height = height;
        this.width = width;
        this.channels = channels;
    }

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     * @param centerCropIfNeeded to crop before rescaling and converting
     */
    public NativeImageLoader(long height, long width, long channels, boolean centerCropIfNeeded) {
        this(height, width, channels);
        this.centerCropIfNeeded = centerCropIfNeeded;
    }

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     * @param imageTransform to use before rescaling and converting
     */
    public NativeImageLoader(long height, long width, long channels, ImageTransform imageTransform) {
        this(height, width, channels);
        this.imageTransform = imageTransform;
    }
    
    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     * @param mode how to load multipage image
     */
    public NativeImageLoader(long height, long width, long channels, MultiPageMode mode) {
        this(height, width, channels);
        this.multiPageMode = mode;
    }

    protected NativeImageLoader(NativeImageLoader other) {
        this.height = other.height;
        this.width = other.width;
        this.channels = other.channels;
        this.centerCropIfNeeded = other.centerCropIfNeeded;
        this.imageTransform = other.imageTransform;
        this.multiPageMode = other.multiPageMode;
    }

    @Override
    public String[] getAllowedFormats() {
        return ALLOWED_FORMATS;
    }

    /**
     * Convert a file to a row vector
     *
     * @param f the image to convert
     * @return the flattened image
     * @throws IOException
     */
    @Override
    public INDArray asRowVector(File f) throws IOException {
        return asMatrix(f).ravel();
    }

    @Override
    public INDArray asRowVector(InputStream is) throws IOException {
        return asMatrix(is).ravel();
    }

    /**
     * Returns {@code asMatrix(image).ravel()}.
     * @see #asMatrix(Object)
     */
    public INDArray asRowVector(Object image) throws IOException {
        return asMatrix(image).ravel();
    }

    public INDArray asRowVector(Frame image) throws IOException {
        return asMatrix(image).ravel();
    }

    public INDArray asRowVector(Mat image) throws IOException {
        return asMatrix(image).ravel();
    }

    static Mat convert(PIX pix) {
        PIX tempPix = null;
        if (pix.colormap() != null) {
            PIX pix2 = pixRemoveColormap(pix, REMOVE_CMAP_TO_FULL_COLOR);
            tempPix = pix = pix2;
        } else if (pix.d() < 8) {
            PIX pix2 = null;
            switch (pix.d()) {
                case 1:
                    pix2 = pixConvert1To8(null, pix, (byte) 0, (byte) 255);
                    break;
                case 2:
                    pix2 = pixConvert2To8(pix, (byte) 0, (byte) 85, (byte) 170, (byte) 255, 0);
                    break;
                case 4:
                    pix2 = pixConvert4To8(pix, 0);
                    break;
                default:
                    assert false;
            }
            tempPix = pix = pix2;
        }
        int height = pix.h();
        int width = pix.w();
        int channels = pix.d() / 8;
        Mat mat = new Mat(height, width, CV_8UC(channels), pix.data(), 4 * pix.wpl());
        Mat mat2 = new Mat(height, width, CV_8UC(channels));
        // swap bytes if needed
        int[] swap = {0, channels - 1, 1, channels - 2, 2, channels - 3, 3, channels - 4},
                        copy = {0, 0, 1, 1, 2, 2, 3, 3},
                        fromTo = channels > 1 && ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN) ? swap : copy;
        mixChannels(mat, 1, mat2, 1, fromTo, Math.min(channels, fromTo.length / 2));
        if (tempPix != null) {
            pixDestroy(tempPix);
        }
        return mat2;
    }


    @Override
    public INDArray asMatrix(File f) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            return asMatrix(bis);
        }
    }

    @Override
    public INDArray asMatrix(InputStream is) throws IOException {
        Mat mat = streamToMat(is);
        INDArray a;
        if (this.multiPageMode != null) {
             a = asMatrix(mat.data(), mat.cols());
        }else{
            Mat image = imdecode(mat, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            if (image == null || image.empty()) {
                PIX pix = pixReadMem(mat.data(), mat.cols());
                if (pix == null) {
                    throw new IOException("Could not decode image from input stream");
                }
                image = convert(pix);
                pixDestroy(pix);
            }
            a = asMatrix(image);
            image.deallocate();
        }
        return a;
    }

    /**
     * Read the stream to the buffer, and return the number of bytes read
     * @param is Input stream to read
     * @return Mat with the buffer data as a row vector
     * @throws IOException
     */
    private Mat streamToMat(InputStream is) throws IOException {
        if(buffer == null){
            buffer = IOUtils.toByteArray(is);
            bufferMat = new Mat(buffer);
            return bufferMat;
        } else {
            int numReadTotal = is.read(buffer);
            //Need to know if all data has been read.
            //(a) if numRead < buffer.length - got everything
            //(b) if numRead >= buffer.length: we MIGHT have got everything (exact right size buffer) OR we need more data

            if(numReadTotal < buffer.length){
                bufferMat.data().put(buffer, 0, numReadTotal);
                bufferMat.cols(numReadTotal);
                return bufferMat;
            }

            //Buffer is full; reallocate and keep reading
            int numReadCurrent = numReadTotal;
            while(numReadCurrent != -1){
                byte[] oldBuffer = buffer;
                if(oldBuffer.length == Integer.MAX_VALUE){
                    throw new IllegalStateException("Cannot read more than Integer.MAX_VALUE bytes");
                }
                //Double buffer, but allocate at least 1MB more
                long increase = Math.max(buffer.length, MIN_BUFFER_STEP_SIZE);
                int newBufferLength = (int)Math.min(Integer.MAX_VALUE, buffer.length + increase);

                buffer = new byte[newBufferLength];
                System.arraycopy(oldBuffer, 0, buffer, 0, oldBuffer.length);
                numReadCurrent = is.read(buffer, oldBuffer.length, buffer.length - oldBuffer.length);
                if(numReadCurrent > 0){
                    numReadTotal += numReadCurrent;
                }
            }

            bufferMat = new Mat(buffer);
            return bufferMat;
        }

    }

    @Override
    public Image asImageMatrix(File f) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            return asImageMatrix(bis);
        }
    }

    @Override
    public Image asImageMatrix(InputStream is) throws IOException {
        Mat mat = streamToMat(is);
        Mat image = imdecode(mat, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        if (image == null || image.empty()) {
            PIX pix = pixReadMem(mat.data(), mat.cols());
            if (pix == null) {
                throw new IOException("Could not decode image from input stream");
            }
            image = convert(pix);
            pixDestroy(pix);
        }
        INDArray a = asMatrix(image);
        Image i = new Image(a, image.channels(), image.rows(), image.cols());

        image.deallocate();
        return i;
    }

    /**
     * Calls {@link AndroidNativeImageLoader#asMatrix(android.graphics.Bitmap)} or
     * {@link Java2DNativeImageLoader#asMatrix(java.awt.image.BufferedImage)}.
     * @param image as a {@link android.graphics.Bitmap} or {@link java.awt.image.BufferedImage}
     * @return the matrix or null for unsupported object classes
     * @throws IOException
     */
    public INDArray asMatrix(Object image) throws IOException {
        INDArray array = null;
        if (array == null) {
            try {
                array = new AndroidNativeImageLoader(this).asMatrix(image);
            } catch (NoClassDefFoundError e) {
                // ignore
            }
        }
        if (array == null) {
            try {
                array = new Java2DNativeImageLoader(this).asMatrix(image);
            } catch (NoClassDefFoundError e) {
                // ignore
            }
        }
        return array;
    }


    protected void fillNDArray(Mat image, INDArray ret) {
        long rows = image.rows();
        long cols = image.cols();
        long channels = image.channels();

        if (ret.lengthLong() != rows * cols * channels) {
            throw new ND4JIllegalStateException("INDArray provided to store image not equal to image: {channels: "
                            + channels + ", rows: " + rows + ", columns: " + cols + "}");
        }

        Indexer idx = image.createIndexer(direct);
        Pointer pointer = ret.data().pointer();
        long[] stride = ret.stride();
        boolean done = false;
        PagedPointer pagedPointer = new PagedPointer(pointer, rows * cols * channels,
                        ret.data().offset() * Nd4j.sizeOfDataType(ret.data().dataType()));

        if (pointer instanceof FloatPointer) {
            FloatIndexer retidx = FloatIndexer.create((FloatPointer) pagedPointer.asFloatPointer(),
                            new long[] {channels, rows, cols}, new long[] {stride[0], stride[1], stride[2]}, direct);
            if (idx instanceof UByteIndexer) {
                UByteIndexer ubyteidx = (UByteIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, ubyteidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            } else if (idx instanceof UShortIndexer) {
                UShortIndexer ushortidx = (UShortIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, ushortidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            } else if (idx instanceof IntIndexer) {
                IntIndexer intidx = (IntIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, intidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            } else if (idx instanceof FloatIndexer) {
                FloatIndexer floatidx = (FloatIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, floatidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            }
            retidx.release();
        } else if (pointer instanceof DoublePointer) {
            DoubleIndexer retidx = DoubleIndexer.create((DoublePointer) pagedPointer.asDoublePointer(),
                            new long[] {channels, rows, cols}, new long[] {stride[0], stride[1], stride[2]}, direct);
            if (idx instanceof UByteIndexer) {
                UByteIndexer ubyteidx = (UByteIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, ubyteidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            } else if (idx instanceof UShortIndexer) {
                UShortIndexer ushortidx = (UShortIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, ushortidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            } else if (idx instanceof IntIndexer) {
                IntIndexer intidx = (IntIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, intidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            } else if (idx instanceof FloatIndexer) {
                FloatIndexer floatidx = (FloatIndexer) idx;
                for (long k = 0; k < channels; k++) {
                    for (long i = 0; i < rows; i++) {
                        for (long j = 0; j < cols; j++) {
                            retidx.put(k, i, j, floatidx.get(i, j, k));
                        }
                    }
                }
                done = true;
            }
            retidx.release();
        }


        if (!done) {
            for (long k = 0; k < channels; k++) {
                for (long i = 0; i < rows; i++) {
                    for (long j = 0; j < cols; j++) {
                        if (channels > 1) {
                            ret.putScalar(k, i, j, idx.getDouble(i, j, k));
                        } else {
                            ret.putScalar(i, j, idx.getDouble(i, j));
                        }
                    }
                }
            }
        }

        idx.release();
        image.data();
        Nd4j.getAffinityManager().tagLocation(ret, AffinityManager.Location.HOST);
    }

    public void asMatrixView(InputStream is, INDArray view) throws IOException {
        Mat mat = streamToMat(is);
        Mat image = imdecode(mat, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        if (image == null || image.empty()) {
            PIX pix = pixReadMem(mat.data(), mat.cols());
            if (pix == null) {
                throw new IOException("Could not decode image from input stream");
            }
            image = convert(pix);
            pixDestroy(pix);
        }
        if (image == null)
            throw new RuntimeException();
        asMatrixView(image, view);
        image.deallocate();
    }

    public void asMatrixView(File f, INDArray view) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            asMatrixView(bis, view);
        }
    }

    public void asMatrixView(Mat image, INDArray view) throws IOException {
        transformImage(image, view);
    }

    public INDArray asMatrix(Frame image) throws IOException {
        return asMatrix(converter.convert(image));
    }

    public INDArray asMatrix(Mat image) throws IOException {
        INDArray ret = transformImage(image, null);

        return ret.reshape(ArrayUtil.combine(new long[] {1}, ret.shape()));
    }

    protected INDArray transformImage(Mat image, INDArray ret) throws IOException {
        if (imageTransform != null && converter != null) {
            ImageWritable writable = new ImageWritable(converter.convert(image));
            writable = imageTransform.transform(writable);
            image = converter.convert(writable.getFrame());
        }
        Mat image2 = null, image3 = null, image4 = null;

        if (channels > 0 && image.channels() != channels) {
            int code = -1;
            switch (image.channels()) {
                case 1:
                    switch ((int)channels) {
                        case 3:
                            code = CV_GRAY2BGR;
                            break;
                        case 4:
                            code = CV_GRAY2RGBA;
                            break;
                    }
                    break;
                case 3:
                    switch ((int)channels) {
                        case 1:
                            code = CV_BGR2GRAY;
                            break;
                        case 4:
                            code = CV_BGR2RGBA;
                            break;
                    }
                    break;
                case 4:
                    switch ((int)channels) {
                        case 1:
                            code = CV_RGBA2GRAY;
                            break;
                        case 3:
                            code = CV_RGBA2BGR;
                            break;
                    }
                    break;
            }
            if (code < 0) {
                throw new IOException("Cannot convert from " + image.channels() + " to " + channels + " channels.");
            }
            image2 = new Mat();
            cvtColor(image, image2, code);
            image = image2;
        }
        if (centerCropIfNeeded) {
            image3 = centerCropIfNeeded(image);
            if (image3 != image) {
                image = image3;
            } else {
                image3 = null;
            }
        }
        image4 = scalingIfNeed(image);
        if (image4 != image) {
            image = image4;
        } else {
            image4 = null;
        }

        if (ret == null) {
            int rows = image.rows();
            int cols = image.cols();
            int channels = image.channels();
            ret = Nd4j.create(channels, rows, cols);
            Nd4j.getMemoryManager().memset(ret);
        }
        fillNDArray(image, ret);

        image.data(); // dummy call to make sure it does not get deallocated prematurely
        if (image2 != null) {
            image2.deallocate();
        }
        if (image3 != null) {
            image3.deallocate();
        }
        if (image4 != null) {
            image4.deallocate();
        }
        return ret;
    }

    // TODO build flexibility on where to crop the image
    protected Mat centerCropIfNeeded(Mat img) {
        int x = 0;
        int y = 0;
        int height = img.rows();
        int width = img.cols();
        int diff = Math.abs(width - height) / 2;

        if (width > height) {
            x = diff;
            width = width - diff;
        } else if (height > width) {
            y = diff;
            height = height - diff;
        }
        return img.apply(new Rect(x, y, width, height));
    }

    protected Mat scalingIfNeed(Mat image) {
        return scalingIfNeed(image, height, width);
    }

    protected Mat scalingIfNeed(Mat image, long dstHeight, long dstWidth) {
        Mat scaled = image;
        if (dstHeight > 0 && dstWidth > 0 && (image.rows() != dstHeight || image.cols() != dstWidth)) {
            resize(image, scaled = new Mat(), new Size(
                    (int)Math.min(dstWidth, Integer.MAX_VALUE),
                    (int)Math.min(dstHeight, Integer.MAX_VALUE)));
        }
        return scaled;
    }


    /**
     * Convert a file to a INDArray
     *
     * @param f the image to convert
     * @return INDArray
     * @throws IOException
     */
    public ImageWritable asWritable(File f) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            Mat mat = streamToMat(bis);
            Mat image = imdecode(mat, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            if (image == null || image.empty()) {
                PIX pix = pixReadMem(mat.data(), mat.cols());
                if (pix == null) {
                    throw new IOException("Could not decode image from input stream");
                }
                image = convert(pix);
                pixDestroy(pix);
            }

            ImageWritable writable = new ImageWritable(converter.convert(image));
            return writable;
        }
    }

    /**
     * Convert ImageWritable to INDArray
     *
     * @param writable ImageWritable to convert
     * @return INDArray
     * @throws IOException
     */
    public INDArray asMatrix(ImageWritable writable) throws IOException {
        Mat image = converter.convert(writable.getFrame());
        return asMatrix(image);
    }

    /** Returns {@code asFrame(array, -1)}. */
    public Frame asFrame(INDArray array) {
        return converter.convert(asMat(array));
    }

    /**
     * Converts an INDArray to a JavaCV Frame. Only intended for images with rank 3.
     *
     * @param array to convert
     * @param dataType from JavaCV (DEPTH_FLOAT, DEPTH_UBYTE, etc), or -1 to use same type as the INDArray
     * @return data copied to a Frame
     */
    public Frame asFrame(INDArray array, int dataType) {
        return converter.convert(asMat(array, OpenCVFrameConverter.getMatDepth(dataType)));
    }

    /** Returns {@code asMat(array, -1)}. */
    public Mat asMat(INDArray array) {
        return asMat(array, -1);
    }

    /**
     * Converts an INDArray to an OpenCV Mat. Only intended for images with rank 3.
     *
     * @param array to convert
     * @param dataType from OpenCV (CV_32F, CV_8U, etc), or -1 to use same type as the INDArray
     * @return data copied to a Mat
     */
    public Mat asMat(INDArray array, int dataType) {
        if (array.rank() > 4 || (array.rank() > 3 && array.size(0) != 1)) {
            throw new UnsupportedOperationException("Only rank 3 (or rank 4 with size(0) == 1) arrays supported");
        }
        int rank = array.rank();
        long[] stride = array.stride();
        long offset = array.data().offset();
        Pointer pointer = array.data().pointer().position(offset);

        long rows = array.size(rank == 3 ? 1 : 2);
        long cols = array.size(rank == 3 ? 2 : 3);
        long channels = array.size(rank == 3 ? 0 : 1);
        boolean done = false;

        if (dataType < 0) {
            dataType = pointer instanceof DoublePointer ? CV_64F : CV_32F;
        }
        Mat mat = new Mat((int)Math.min(rows, Integer.MAX_VALUE), (int)Math.min(cols, Integer.MAX_VALUE),
                CV_MAKETYPE(dataType, (int)Math.min(channels, Integer.MAX_VALUE)));
        Indexer matidx = mat.createIndexer(direct);

        Nd4j.getAffinityManager().ensureLocation(array, AffinityManager.Location.HOST);

        if (pointer instanceof FloatPointer && dataType == CV_32F) {
            FloatIndexer ptridx = FloatIndexer.create((FloatPointer)pointer, new long[] {channels, rows, cols},
                    new long[] {stride[rank == 3 ? 0 : 1], stride[rank == 3 ? 1 : 2], stride[rank == 3 ? 2 : 3]}, direct);
            FloatIndexer idx = (FloatIndexer)matidx;
            for (long k = 0; k < channels; k++) {
                for (long i = 0; i < rows; i++) {
                    for (long j = 0; j < cols; j++) {
                        idx.put(i, j, k, ptridx.get(k, i, j));
                    }
                }
            }
            done = true;
            ptridx.release();
        } else if (pointer instanceof DoublePointer && dataType == CV_64F) {
            DoubleIndexer ptridx = DoubleIndexer.create((DoublePointer)pointer, new long[] {channels, rows, cols},
                    new long[] {stride[rank == 3 ? 0 : 1], stride[rank == 3 ? 1 : 2], stride[rank == 3 ? 2 : 3]}, direct);
            DoubleIndexer idx = (DoubleIndexer)matidx;
            for (long k = 0; k < channels; k++) {
                for (long i = 0; i < rows; i++) {
                    for (long j = 0; j < cols; j++) {
                        idx.put(i, j, k, ptridx.get(k, i, j));
                    }
                }
            }
            done = true;
            ptridx.release();
        }

        if (!done) {
            for (long k = 0; k < channels; k++) {
                for (long i = 0; i < rows; i++) {
                    for (long j = 0; j < cols; j++) {
                        if (rank == 3) {
                            matidx.putDouble(new long[] {i, j, k}, array.getDouble(k, i, j));
                        } else {
                            matidx.putDouble(new long[] {i, j, k}, array.getDouble(0, k, i, j));
                        }
                    }
                }
            }
        }

        matidx.release();
        return mat;
    }

    /**
     * Read multipage tiff and load into INDArray
     *
     * @param bytes
     * @return INDArray
     * @throws IOException
     */
    private INDArray asMatrix(BytePointer bytes, long length) throws IOException {
        PIXA pixa;
        pixa = pixaReadMemMultipageTiff(bytes, length);
        INDArray data;
        INDArray currentD;
        INDArrayIndex[] index = null;
        switch (this.multiPageMode) {
            case MINIBATCH:
                data = Nd4j.create(pixa.n(), 1, pixa.pix(0).h(), pixa.pix(0).w());
                break;
            case CHANNELS:
                data = Nd4j.create(1, pixa.n(), pixa.pix(0).h(), pixa.pix(0).w());
                break;
            case FIRST:
                data = Nd4j.create(1, 1, pixa.pix(0).h(), pixa.pix(0).w());
                PIX pix = pixa.pix(0);
                currentD = asMatrix(convert(pix));
                pixDestroy(pix);
                index = new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all()};
                data.put(index , currentD.get(NDArrayIndex.all(), NDArrayIndex.all(),NDArrayIndex.all()));
                return data;
            default: throw new UnsupportedOperationException("Unsupported MultiPageMode: " + multiPageMode);
        }
        for (int i = 0; i < pixa.n(); i++) {
            PIX pix = pixa.pix(i);
            currentD = asMatrix(convert(pix));
            pixDestroy(pix);
            //TODO to change when 16-bit image is supported
            switch (this.multiPageMode) {
                case MINIBATCH:
                    index = new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all()};
                    break;
                case CHANNELS:
                    index = new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(i),NDArrayIndex.all(),NDArrayIndex.all()};
                    break;
                default: throw new UnsupportedOperationException("Unsupported MultiPageMode: " + multiPageMode);
            }
            data.put(index , currentD.get(NDArrayIndex.all(), NDArrayIndex.all(),NDArrayIndex.all()));
        }

        return data;
    }
    
}
