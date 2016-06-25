package org.nd4j.linalg.dataset;

import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

/**
 * Read in existing mini batches created
 * by the mini batch file datasetiterator.
 *
 * @author Adam Gibson
 */
public class ExistingMiniBatchDataSetIterator implements DataSetIterator {
    private List<String[]> paths;
    private int currIdx;
    private File rootDir;
    private int totalBatches = -1;
    private DataSetPreProcessor dataSetPreProcessor;

    /**
     * Create with the given root directory
     * @param rootDir the root directory to use
     */
    public ExistingMiniBatchDataSetIterator(File rootDir) {
        this.rootDir = rootDir;
        this.paths = new ArrayList<>();
        if(totalBatches < 1)
            totalBatches = rootDir.list().length;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Unable to load custom number of examples");
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void reset() {
        currIdx = 0;
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int cursor() {
        return currIdx;
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return currIdx < totalBatches;
    }

    @Override
    public void remove() {
        //no opt;
    }

    @Override
    public DataSet next() {
        try {
            DataSet ret =  read(currIdx);
            if(dataSetPreProcessor != null)
                dataSetPreProcessor.preProcess(ret);
            currIdx++;

            return ret;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read dataset");
        }
    }

    private DataSet read(int idx) throws IOException {
        File path = new File(rootDir,String.format("dataset-%d.bin",idx));
        DataSet d = new DataSet();
        d.load(path);
        return d;
    }
}
