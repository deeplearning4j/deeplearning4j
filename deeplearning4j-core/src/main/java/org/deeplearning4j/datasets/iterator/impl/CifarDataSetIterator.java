package org.deeplearning4j.datasets.iterator.impl;

import org.canova.image.loader.CifarLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.*;

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
    protected int totalExamples = CifarLoader.NUM_TRAIN_IMAGES + CifarLoader.NUM_TRAIN_IMAGES;
    // TODO use maxNumBatches and batchNum instead
    protected int numExamples = totalExamples;
    protected int exampleCount = 0;

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples) {
        super(null, batchSize, 1 , CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader();
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, String version) {
        super(null, batchSize, 1, CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader(version);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
    }


    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
        super(null, batchSize, 1, CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2]);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, String version) {
        super(null, batchSize, 1, CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2], version);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories) {
        super(null, batchSize, 1, numCategories);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2]);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, String version) {
        super(null, batchSize, 1, numCategories);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2], version);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param numExamples the overall number of examples
     * @param numCategories the overall number of labels
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int numCategories) {
        super(null, batchSize, 1, numCategories);
        this.loader = new CifarLoader();
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;

    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     */
    public CifarDataSetIterator(int batchSize, int[] imgDim)  {
        super(null, batchSize, 1, CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2]);
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
        INDArray label = null; // first value in the 3073 byte array
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
                dataSets.add(new DataSet(loader.asRowVector(image), label));
                batchNumCount++;
            }
            exampleCount += batchSize;
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for (DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }

        if(inputs.isEmpty() || (maxNumBatches > -1 && batchNum >= maxNumBatches)) {
            notOvershot = false;
            return last;
        }

        DataSet ret =  new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
        last = ret;
        if(preProcessor != null) preProcessor.preProcess(ret);
        if ( loader.getLabels() != null) ret.setLabelNames(loader.getLabels());
        return ret;
    }

    @Override
    public boolean hasNext() {
        try {
            return exampleCount < numExamples;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    @Override
    public int totalExamples() {
        return totalExamples;
    }

    @Override
    public void reset() {
        inputStream = loader.getInputStream();
    }

    @Override
    public List<String> getLabels(){
        return loader.getLabels();
    }


}
