package org.deeplearning4j.datasets.iterator.impl;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_8UC;
import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Created by nyghtowl on 12/18/15.
 */
public class CifarDataSetIterator extends RecordReaderDataSetIterator {

    protected static int height = 32;
    protected static int width = 32;
    protected static int channels = 3;
    protected static int numPixels = 3073;
    protected static CifarLoader loader;
    protected static InputStream inputStream = null;
    protected int totalExamples = CifarLoader.NUM_TRAIN_IMAGES;
    // TODO use maxNumBatches and batchNum instead
    protected int numExamples = totalExamples;
    protected int exampleCount = 0;
    protected boolean overshot = false;


    /** Loads images with given  batchSize, numExamples, & version returned by the generator. */
    public CifarDataSetIterator(int batchSize, int numExamples,  boolean train) {
        this(batchSize, numExamples, new int[]{height, width, channels}, CifarLoader.NUM_LABELS, null, 0, train);
    }

    /** Loads images with given  batchSize, numExamples, & imgDim returned by the generator. */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
        this(batchSize, numExamples, imgDim, CifarLoader.NUM_LABELS, null, 0, true);
    }

    /** Loads images with given  batchSize, numExamples, imgDim & version returned by the generator. */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, boolean train) {
        this(batchSize, numExamples, imgDim, CifarLoader.NUM_LABELS, null, 0, train);
    }

    /** Loads images with given  batchSize & numExamples returned by the generator. */
    public CifarDataSetIterator(int batchSize, int numExamples) {
        this(batchSize, numExamples, new int[]{height, width, channels}, CifarLoader.NUM_LABELS, null, 0, true);
    }

    /** Loads images with given  batchSize & imgDim returned by the generator. */
    public CifarDataSetIterator(int batchSize, int[] imgDim)  {
        this(batchSize, CifarLoader.NUM_TRAIN_IMAGES, imgDim, CifarLoader.NUM_LABELS, null, 0, true);
    }

    /** Loads images with given  batchSize, numExamples, imgDim & version returned by the generator. */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int normalizeValue, boolean train) {
        this(batchSize, numExamples, imgDim, CifarLoader.NUM_LABELS, null, normalizeValue, train);
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     * @param numExamples the overall number of examples
     * @param imageTransform the transformation to apply to the images
     * @param normalizeValue value to normalize the image data
     * @param train true if use training set and false for test
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numPossibleLables, ImageTransform imageTransform, int normalizeValue, boolean train) {
        super(null, batchSize, 1, numExamples);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2], imageTransform, normalizeValue, train);
        this.totalExamples = train? totalExamples: CifarLoader.NUM_TEST_IMAGES;
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
        this.numPossibleLabels = numPossibleLables;
        this.inputStream  = loader.getInputStream();
    }

    @Override
    public DataSet next(int num) {
        if(useCurrent) {
            useCurrent = false;
            if(preProcessor != null) preProcessor.preProcess(last);
            return last;
        }

        int batchNumCount = 0;
        byte[] byteFeature = new byte[numPixels];
        List<DataSet> dataSets = new ArrayList<>();
        INDArray label; // first value in the 3073 byte array
        Mat image = new Mat(height, width, CV_8UC(channels)); // feature are 3072
        ByteBuffer imageData = image.createBuffer();

        try {
            while((inputStream.read(byteFeature)) != -1 && batchNumCount != num) {
                label = FeatureUtil.toOutcomeVector(byteFeature[0], numPossibleLabels);
                for (int i = 0; i < height * width; i++) {
                    imageData.put(3 * i,     byteFeature[i + 1 + 2 * height * width]); // blue
                    imageData.put(3 * i + 1, byteFeature[i + 1 +     height * width]); // green
                    imageData.put(3 * i + 2, byteFeature[i + 1                     ]); // red
                }
                try {
                    dataSets.add(new DataSet(loader.asRowVector(image), label));
                    batchNumCount++;
                } catch(Exception e){
                    break;
                }
            }
            exampleCount += batchNumCount;
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for (DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }

        batchNum++;
        if(inputs.isEmpty() || (maxNumBatches > -1 && batchNum >= maxNumBatches)) {
            overshot = true;
            return last;
        }

        DataSet ret =  new DataSet(Nd4j.vstack(inputs), Nd4j.vstack(labels));
        last = ret;
        if(preProcessor != null) preProcessor.preProcess(ret);
        if ( loader.getLabels() != null) ret.setLabelNames(loader.getLabels());
        return ret;
    }

    @Override
    public boolean hasNext() {
        return exampleCount < numExamples && (maxNumBatches == -1 || batchNum < maxNumBatches) && !overshot;
    }

    @Override
    public int totalExamples() {
        return totalExamples;
    }

    @Override
    public void reset() {
        exampleCount = 0;
        overshot = false;
        batchNum = 0;
        inputStream = loader.getInputStream();
    }

    @Override
    public List<String> getLabels(){
        return loader.getLabels();
    }


}
