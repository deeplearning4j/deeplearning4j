package org.deeplearning4j.datasets.iterator.file;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.List;
import java.util.Random;

public class FileDataSetIterator extends BaseFileIterator<DataSet,DataSetPreProcessor> implements DataSetIterator {

    /**
     * Create a FileDataSetIterator with the following default settings:
     * - Recursive: files in subdirectories are included
     * - Randomization: order of examples is randomized with a random RNG seed
     * - Batch size: default (as in the stored DataSets - no splitting/combining)
     * - File extensions: no filtering - all files in directory are assumed to be a DataSet
     *
     * @param rootDir Root directory containing the D
     */
    public FileDataSetIterator(File rootDir){
        this(rootDir, true, new Random(), -1, (String[])null);
    }

    public FileDataSetIterator(File rootDir, int batchSize){
        this(rootDir, batchSize, (String[])null);
    }

    public FileDataSetIterator(File rootDir, String... validExtensions){
        super(rootDir, -1, validExtensions);
    }

    public FileDataSetIterator(File rootDir, int batchSize, String... validExtensions){
        super(rootDir, batchSize, validExtensions);
    }

    public FileDataSetIterator(File rootDir, boolean recursive, Random rng, int batchSize, String... validExtensions){
        super(rootDir, recursive, rng, batchSize, validExtensions);
    }

    @Override
    protected DataSet load(File f) {
        DataSet ds = new DataSet();
        ds.load(f);
        return ds;
    }

    @Override
    protected int sizeOf(DataSet of) {
        return of.numExamples();
    }

    @Override
    protected List<DataSet> split(DataSet toSplit) {
        return toSplit.asList();
    }

    @Override
    protected DataSet merge(List<DataSet> toMerge) {
        return DataSet.merge(toMerge);
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Not supported for this iterator");
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Not supported for this iterator");
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException("Not supported for this iterator");
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException("Not supported for this iterator");
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return position;
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException("Not supported for this iterator");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not supported for this iterator");
    }
}
