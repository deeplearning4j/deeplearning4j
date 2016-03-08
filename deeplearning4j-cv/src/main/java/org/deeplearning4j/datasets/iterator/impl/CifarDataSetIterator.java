package org.deeplearning4j.datasets.iterator.impl;

import org.canova.image.loader.CifarLoader;
import org.canova.image.loader.ImageLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by nyghtowl on 12/18/15.
 */
public class CifarDataSetIterator extends RecordReaderDataSetIterator {

    protected static int numPixels = 3073;
    protected  CifarLoader loader;
    protected ImageLoader realLoader;
    protected  InputStream inputStream = null;
    protected int totalExamples = CifarLoader.NUM_TRAIN_IMAGES + CifarLoader.NUM_TRAIN_IMAGES;
    protected int numExamples = totalExamples;
    protected int exampleCount = 0;

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples) {
        super(null, batchSize, CifarLoader.WIDTH * CifarLoader.HEIGHT * CifarLoader.CHANNELS, CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader();
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples ? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, String version) {
        super(null, batchSize, CifarLoader.WIDTH * CifarLoader.HEIGHT * CifarLoader.CHANNELS, CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader(version);
        this.realLoader = new ImageLoader( CifarLoader.WIDTH,CifarLoader.HEIGHT,CifarLoader.CHANNELS);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples ? totalExamples: numExamples;
    }


    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
        super(null, batchSize, imgDim[0] * imgDim[1] * imgDim[2], CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader();
        realLoader = new ImageLoader(imgDim[0],imgDim[1],imgDim[2]);
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples ? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories) {
        super(null, batchSize, imgDim[0] * imgDim[1] * imgDim[2], numCategories);
        this.loader = new CifarLoader();
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples ? totalExamples: numExamples;
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param numCategories the overall number of labels
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int numCategories) {
        super(null, batchSize, CifarLoader.WIDTH * CifarLoader.HEIGHT * CifarLoader.CHANNELS, numCategories);
        this.loader = new CifarLoader();
        this.inputStream  = loader.getInputStream();
        this.numExamples = numExamples > totalExamples? totalExamples: numExamples;

    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     */
    public CifarDataSetIterator(int batchSize, int[] imgDim)  {
        super(null, batchSize, imgDim[0] * imgDim[1] * imgDim[2], CifarLoader.NUM_LABELS);
        this.loader = new CifarLoader();
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
        INDArray featureVector; // feature are 3072
        try {
            while((inputStream.read(byteFeature)) != -1 && batchNumCount != num) {
                label = FeatureUtil.toOutcomeVector(byteFeature[0], numPossibleLabels);
                ByteArrayInputStream bis = new ByteArrayInputStream(byteFeature);
                bis.skip(1);
                //passively auto converts to the proper size image based on the initialized shape.
                featureVector =  realLoader.asMatrix(bis);
                dataSets.add(new DataSet(featureVector, label));
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

        if(inputs.isEmpty()) {
            overshot = true;
            return last;
        }

        DataSet ret =  new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
        last = ret;
        if(preProcessor != null) preProcessor.preProcess(ret);
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
        try {
            inputStream.reset();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public List<String> getLabels(){
        return loader.getLabels();
    }


}
