/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.image.recordreader.objdetect;

import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.util.ImageUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.util.files.URIUtil;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.image.transform.ImageTransform;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * An image record reader for object detection.
 * <p>
 * Format of returned values: 4d array, with dimensions [minibatch, 4+C, h, w]
 * Where the image is quantized into h x w grid locations.
 * <p>
 * Note that this matches the format required for Deeplearning4j's Yolo2OutputLayer
 *
 * @author Alex Black
 */
public class ObjectDetectionRecordReader extends BaseImageRecordReader {

    private final int gridW;
    private final int gridH;
    private final ImageObjectLabelProvider labelProvider;

    protected Image currentImage;

    /**
     *
     * @param height        Height of the output images
     * @param width         Width of the output images
     * @param channels      Number of channels for the output images
     * @param gridH         Grid/quantization size (along  height dimension) - Y axis
     * @param gridW         Grid/quantization size (along  height dimension) - X axis
     * @param labelProvider ImageObjectLabelProvider - used to look up which objects are in each image
     */
    public ObjectDetectionRecordReader(int height, int width, int channels, int gridH, int gridW, ImageObjectLabelProvider labelProvider) {
        super(height, width, channels, null, null);
        this.gridW = gridW;
        this.gridH = gridH;
        this.labelProvider = labelProvider;
        this.appendLabel = labelProvider != null;
    }

    /**
     * When imageTransform != null, object is removed if new center is outside of transformed image bounds.
     *
     * @param height        Height of the output images
     * @param width         Width of the output images
     * @param channels      Number of channels for the output images
     * @param gridH         Grid/quantization size (along  height dimension) - Y axis
     * @param gridW         Grid/quantization size (along  height dimension) - X axis
     * @param labelProvider ImageObjectLabelProvider - used to look up which objects are in each image
     * @param imageTransform ImageTransform - used to transform image and coordinates
     */
    public ObjectDetectionRecordReader(int height, int width, int channels, int gridH, int gridW,
            ImageObjectLabelProvider labelProvider, ImageTransform imageTransform) {
        super(height, width, channels, null, null);
        this.gridW = gridW;
        this.gridH = gridH;
        this.labelProvider = labelProvider;
        this.appendLabel = labelProvider != null;
        this.imageTransform = imageTransform;
    }

    @Override
    public List<Writable> next() {
        return next(1).get(0);
    }

    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
        inputSplit = split;
        URI[] locations = split.locations();
        Set<String> labelSet = new HashSet<>();
        if (locations != null && locations.length >= 1) {
            for (URI location : locations) {
                List<ImageObject> imageObjects = labelProvider.getImageObjectsForPath(location);
                for (ImageObject io : imageObjects) {
                    String name = io.getLabel();
                    if (!labelSet.contains(name)) {
                        labelSet.add(name);
                    }
                }
            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
        } else {
            throw new IllegalArgumentException("No path locations found in the split.");
        }

        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        labels = new ArrayList<>(labelSet);
        Collections.sort(labels);
    }

    @Override
    public List<List<Writable>> next(int num) {
        List<File> files = new ArrayList<>(num);
        List<List<ImageObject>> objects = new ArrayList<>(num);

        for (int i = 0; i < num && hasNext(); i++) {
            File f = iter.next();
            this.currentFile = f;
            if (!f.isDirectory()) {
                files.add(f);
                objects.add(labelProvider.getImageObjectsForPath(f.getPath()));
            }
        }

        int nClasses = labels.size();

        INDArray outImg = Nd4j.create(files.size(), channels, height, width);
        INDArray outLabel = Nd4j.create(files.size(), 4 + nClasses, gridH, gridW);

        int exampleNum = 0;
        for (int i = 0; i < files.size(); i++) {
            File imageFile = files.get(i);
            this.currentFile = imageFile;
            try {
                this.invokeListeners(imageFile);
                Image image = this.imageLoader.asImageMatrix(imageFile);
                this.currentImage = image;
                Nd4j.getAffinityManager().ensureLocation(image.getImage(), AffinityManager.Location.DEVICE);

                outImg.put(new INDArrayIndex[]{point(exampleNum), all(), all(), all()}, image.getImage());

                List<ImageObject> objectsThisImg = objects.get(exampleNum);

                label(image, objectsThisImg, outLabel, exampleNum);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            exampleNum++;
        }

        return new NDArrayRecordBatch(Arrays.asList(outImg, outLabel));
    }

    private void label(Image image, List<ImageObject> objectsThisImg, INDArray outLabel, int exampleNum) {
        int oW = image.getOrigW();
        int oH = image.getOrigH();

        int W = oW;
        int H = oH;

        //put the label data into the output label array
        for (ImageObject io : objectsThisImg) {
            double cx = io.getXCenterPixels();
            double cy = io.getYCenterPixels();
            if (imageTransform != null) {
                W = imageTransform.getCurrentImage().getWidth();
                H = imageTransform.getCurrentImage().getHeight();

                float[] pts = imageTransform.query(io.getX1(), io.getY1(), io.getX2(), io.getY2());

                int minX = Math.round(Math.min(pts[0], pts[2]));
                int maxX = Math.round(Math.max(pts[0], pts[2]));
                int minY = Math.round(Math.min(pts[1], pts[3]));
                int maxY = Math.round(Math.max(pts[1], pts[3]));

                io = new ImageObject(minX, minY, maxX, maxY, io.getLabel());
                cx = io.getXCenterPixels();
                cy = io.getYCenterPixels();

                if (cx < 0 || cx >= W || cy < 0 || cy >= H) {
                    continue;
                }
            }

            double[] cxyPostScaling = ImageUtils.translateCoordsScaleImage(cx, cy, W, H, width, height);
            double[] tlPost = ImageUtils.translateCoordsScaleImage(io.getX1(), io.getY1(), W, H, width, height);
            double[] brPost = ImageUtils.translateCoordsScaleImage(io.getX2(), io.getY2(), W, H, width, height);

            //Get grid position for image
            int imgGridX = (int) (cxyPostScaling[0] / width * gridW);
            int imgGridY = (int) (cxyPostScaling[1] / height * gridH);

            //Convert pixels to grid position, for TL and BR X/Y
            tlPost[0] = tlPost[0] / width * gridW;
            tlPost[1] = tlPost[1] / height * gridH;
            brPost[0] = brPost[0] / width * gridW;
            brPost[1] = brPost[1] / height * gridH;

            //Put TL, BR into label array:
            Preconditions.checkState(imgGridY >= 0 && imgGridY < outLabel.size(2), "Invalid image center in Y axis: "
                    + "calculated grid location of %s, must be between 0 (inclusive) and %s (exclusive). Object label center is outside "
                    + "of image bounds. Image object: %s", imgGridY, outLabel.size(2), io);
            Preconditions.checkState(imgGridX >= 0 && imgGridX < outLabel.size(3), "Invalid image center in X axis: "
                    + "calculated grid location of %s, must be between 0 (inclusive) and %s (exclusive). Object label center is outside "
                    + "of image bounds. Image object: %s", imgGridY, outLabel.size(2), io);

            outLabel.putScalar(exampleNum, 0, imgGridY, imgGridX, tlPost[0]);
            outLabel.putScalar(exampleNum, 1, imgGridY, imgGridX, tlPost[1]);
            outLabel.putScalar(exampleNum, 2, imgGridY, imgGridX, brPost[0]);
            outLabel.putScalar(exampleNum, 3, imgGridY, imgGridX, brPost[1]);

            //Put label class into label array: (one-hot representation)
            int labelIdx = labels.indexOf(io.getLabel());
            outLabel.putScalar(exampleNum, 4 + labelIdx, imgGridY, imgGridX, 1.0);
        }
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
        Image image = this.imageLoader.asImageMatrix(dataInputStream);
        Nd4j.getAffinityManager().ensureLocation(image.getImage(), AffinityManager.Location.DEVICE);

        List<Writable> ret = RecordConverter.toRecord(image.getImage());
        if (appendLabel) {
            List<ImageObject> imageObjectsForPath = labelProvider.getImageObjectsForPath(uri.getPath());
            int nClasses = labels.size();
            INDArray outLabel = Nd4j.create(1, 4 + nClasses, gridH, gridW);
            label(image, imageObjectsForPath, outLabel, 0);
            ret.add(new NDArrayWritable(outLabel));
        }
        return ret;
    }

    @Override
    public Record nextRecord() {
        List<Writable> list = next();
        URI uri = URIUtil.fileToURI(currentFile);
        return new org.datavec.api.records.impl.Record(list, new RecordMetaDataImageURI(uri, BaseImageRecordReader.class,
                currentImage.getOrigC(), currentImage.getOrigH(), currentImage.getOrigW()));
    }
}
