package org.deeplearning4j.datasets.iterator.impl;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

/**
 *
 */

public class CifarDataSetIterator extends RecordReaderDataSetIterator {

    protected static int height = 32;
    protected static int width = 32;
    protected static int channels = 3;
    protected static CifarLoader loader;
    protected int totalExamples = CifarLoader.NUM_TRAIN_IMAGES;
    // TODO use maxNumBatches and batchNum instead
    protected int numExamples = totalExamples;
    protected int exampleCount = 0;
    protected boolean overshot = false;
    protected ImageTransform imageTransform;
    protected static boolean usePreProcessCifar = false;
    protected static boolean train = true;

    /**
     * Loads images with given  batchSize, numExamples, & version returned by the generator.
     */
    public CifarDataSetIterator(int batchSize, int numExamples, boolean train) {
        this(batchSize, numExamples, new int[]{height, width, channels}, CifarLoader.NUM_LABELS, null, usePreProcessCifar, train);
    }

    /**
     * Loads images with given  batchSize, numExamples, & imgDim returned by the generator.
     */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
        this(batchSize, numExamples, imgDim, CifarLoader.NUM_LABELS, null, usePreProcessCifar, train);
    }

    /**
     * Loads images with given  batchSize, numExamples, imgDim & version returned by the generator.
     */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, boolean train) {
        this(batchSize, numExamples, imgDim, CifarLoader.NUM_LABELS, null, usePreProcessCifar, train);
    }

    /**
     * Loads images with given  batchSize & numExamples returned by the generator.
     */
    public CifarDataSetIterator(int batchSize, int numExamples) {
        this(batchSize, numExamples, new int[]{height, width, channels}, CifarLoader.NUM_LABELS, null, usePreProcessCifar, train);
    }

    /**
     * Loads images with given  batchSize & imgDim returned by the generator.
     */
    public CifarDataSetIterator(int batchSize, int[] imgDim) {
        this(batchSize, CifarLoader.NUM_TRAIN_IMAGES, imgDim, CifarLoader.NUM_LABELS, null, usePreProcessCifar, train);
    }

    /**
     * Loads images with given  batchSize, numExamples, imgDim & version returned by the generator.
     */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, boolean preProcessCifar, boolean train) {
        this(batchSize, numExamples, imgDim, CifarLoader.NUM_LABELS, null, preProcessCifar, train);
    }

    /**
     * Create Cifar data specific iterator
     *
     * @param batchSize      the batch size of the examples
     * @param imgDim         an array of height, width and channels
     * @param numExamples    the overall number of examples
     * @param imageTransform the transformation to apply to the images
     * @param preProcessCifar preprocess cifar
     * @param train          true if use training set and false for test
     */
    public CifarDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numPossibleLables, ImageTransform imageTransform, boolean preProcessCifar, boolean train) {
        super(null, batchSize, 1, numExamples);
        this.loader = new CifarLoader(imgDim[0], imgDim[1], imgDim[2], imageTransform, train, preProcessCifar);
        this.totalExamples = train ? totalExamples : CifarLoader.NUM_TEST_IMAGES;
        this.numExamples = numExamples > totalExamples ? totalExamples : numExamples;
        this.numPossibleLabels = numPossibleLables;
        this.imageTransform = imageTransform;
        this.usePreProcessCifar = preProcessCifar;
        this.train = train;
    }

    // TODO add transform  - random flip when loading batches

    @Override
    public DataSet next(int batchSize) {
        if(useCurrent) {
            useCurrent = false;
            return last;
        }
        DataSet result;
        int i = 1;
        if (usePreProcessCifar) {
            if (train && batchNum == 0 && i <= CifarLoader.TRAINFILENAMES.length) i++;
            result = loader.next(batchSize, i, batchNum);
        }
        else
            result =  loader.next(batchSize);
        exampleCount += batchSize;
        batchNum++;

        if((result == null || result == new DataSet()) || (maxNumBatches > -1 && batchNum >= maxNumBatches)) {
            overshot = true;
            return last;
        }

        if(preProcessor != null) preProcessor.preProcess(result);
        last = result;
        if ( loader.getLabels() != null) result.setLabelNames(loader.getLabels());
        return result;
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
        loader.reset();
    }

    @Override
    public List<String> getLabels(){
        return loader.getLabels();
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    public void train() {
        this.train = true;
        this.loader.train();
        reset();
    }

    public void test() {
        this.train = false;
        this.loader.test();
        reset();
    }


}
