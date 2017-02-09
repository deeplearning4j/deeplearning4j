package org.deeplearning4j.keras.hdf5;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;

/**
 * Iterator reading mini batches of data stored in separate files. Labels and features are expected to be dumped into
 * separate directories. Filenames are expected to adhere to a predefined pattern: `batch_%d.h5`.
 * This class supports only a very narrow subset of the DataSetIterator interface! (e.g.
 * there is no support for pre-processing of data)
 *
 * @author pkoperek@gmail.com
 */
@Slf4j
public class HDF5MiniBatchDataSetIterator implements DataSetIterator {

    private static final String FILE_NAME_PATTERN = "batch_%d.h5";

    private final NDArrayHDF5Reader ndArrayHDF5Reader = new NDArrayHDF5Reader();

    private final File trainFeaturesDirectory;
    private final File trainLabelsDirectory;
    private final int batchesCount;
    private int currentIdx;
    private DataSetPreProcessor preProcessor;
    private int numLabels;


    public HDF5MiniBatchDataSetIterator(String trainFeaturesDirectory) {
        this.trainFeaturesDirectory = new File(trainFeaturesDirectory);
        this.trainLabelsDirectory = null;
        this.batchesCount = this.trainFeaturesDirectory.list().length;
        this.numLabels = -1;
    }


    public HDF5MiniBatchDataSetIterator(String trainFeaturesDirectory, String trainLabelsDirectory) {
        this.trainFeaturesDirectory = new File(trainFeaturesDirectory);
        this.trainLabelsDirectory = new File(trainLabelsDirectory);
        this.batchesCount = this.trainFeaturesDirectory.list().length;

        // infer classes from labels
        String batchFileName = fileNameForIdx(0);
        INDArray labels = ndArrayHDF5Reader.readFromPath(
            Paths.get(this.trainLabelsDirectory.getAbsolutePath(), batchFileName));
        this.numLabels = labels.size(-1);
    }

    @Override
    public boolean hasNext() {
        return currentIdx < batchesCount;
    }

    @Override
    public DataSet next() {
        DataSet dataSet = readIdx(currentIdx);
        currentIdx++;

        if (preProcessor != null) {
            if (!dataSet.isPreProcessed()) {
                preProcessor.preProcess(dataSet);
                dataSet.markAsPreProcessed();
            }
        }

        return dataSet;
    }

    private DataSet readIdx(int currentIdx) {
        String batchFileName = fileNameForIdx(currentIdx);

        if (log.isTraceEnabled()) {
            log.trace("Reading: " + batchFileName);
        }

        if(trainLabelsDirectory!=null) {
            INDArray features = ndArrayHDF5Reader.readFromPath(
                Paths.get(trainFeaturesDirectory.getAbsolutePath(), batchFileName));
            INDArray labels = ndArrayHDF5Reader.readFromPath(
                Paths.get(trainLabelsDirectory.getAbsolutePath(), batchFileName));

            return new DataSet(features, labels);

        } else {
            INDArray features = ndArrayHDF5Reader.readFromPath(
                Paths.get(trainFeaturesDirectory.getAbsolutePath(), batchFileName));

            return new DataSet(features, null);
        }
    }

    private String fileNameForIdx(int currentIdx) {
        return String.format(FILE_NAME_PATTERN, currentIdx);
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        /**
         * Async support is turned off on purpose: otherwise there are indeterministic segfaults in JavaCPP
         * when cleaning memory after HDF5 libs.
         */
        return false;
    }

    @Override
    public void reset() {
        currentIdx = 0;
    }

    @Override
    public int cursor() {
        return currentIdx;
    }


    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Can't load custom number of samples");
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
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        if(numLabels > 0) {
            // generated on-the-fly using known labels dimensions
            List<String> ret = new LinkedList<>();

            for(int i = 0; i < numLabels; i++) {
                ret.add(Integer.toString(i));
            }

            return ret;

        } else {
            throw new IllegalStateException("Number of labels is unknown");
        }

    }

    @Override
    public void remove() {
        // no-op
    }

}
