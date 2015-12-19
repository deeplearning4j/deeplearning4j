package org.deeplearning4j.datasets.iterator.impl;

import org.canova.image.loader.CifarLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

/**
 * Created by nyghtowl on 12/18/15.
 */
public class CifarDataSetIterator extends RecordReaderDataSetIterator {

    protected static int width = 32;
    protected static int height = 32;
    protected static int channels = 3;

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples) {
        super(new CifarLoader().getRecordReader(numExamples), batchSize, width * height * channels, CifarLoader.NUM_LABELS);
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
        super(new CifarLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples), batchSize, imgDim[0] * imgDim[1] * imgDim[2], CifarLoader.NUM_LABELS);
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     * @param numExamples the overall number of examples
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories) {
        super(new CifarLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples), batchSize, imgDim[0] * imgDim[1] * imgDim[2], numCategories);
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param numCategories the overall number of labels
     * */
    public CifarDataSetIterator(int batchSize, int numExamples, int numCategories) {
        super(new CifarLoader().getRecordReader(numExamples, numCategories), batchSize, width * height * channels, numCategories);
    }

    /**
     * Create Cifar data specific iterator
     * @param batchSize the the batch size of the examples
     * @param imgDim an array of width, height and channels
     */
    public CifarDataSetIterator(int batchSize, int[] imgDim)  {
        super(new CifarLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2]), batchSize, imgDim[0] * imgDim[1] * imgDim[2], CifarLoader.NUM_LABELS);
    }

}
