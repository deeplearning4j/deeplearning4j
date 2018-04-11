package org.deeplearning4j.datasets.iterator.file;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.List;
import java.util.Random;

/**
 * Iterate over a directory (and optionally subdirectories) containing a number of {@link DataSet} objects that have
 * previously been saved to files with {@link DataSet#save(File)}.<br>
 * This iterator supports the following (optional) features, depending on the constructor used:<br>
 * - Recursive listing of all files (i.e., include files in subdirectories)<br>
 * - Filtering based on a set of file extensions (if null, no filtering - assume all files are saved DataSet objects)<br>
 * - Randomization of iteration order (default enabled, if a {@link Random} instance is provided<br>
 * - Combining and splitting of DataSets (disabled by default, or if batchSize == -1. If enabled, DataSet objects will
 * be split or combined as required to ensure the specified minibatch size is returned. In other words, the saved
 * DataSet objects can have a different number of examples vs. those returned by the iterator.<br>
 *
 * @author Alex BLack
 */
public class FileDataSetIterator extends BaseFileIterator<DataSet, DataSetPreProcessor> implements DataSetIterator {

    @Getter
    @Setter
    private List<String> labels;

    /**
     * Create a FileDataSetIterator with the following default settings:<br>
     * - Recursive: files in subdirectories are included<br>
     * - Randomization: order of examples is randomized with a random RNG seed<br>
     * - Batch size: default (as in the stored DataSets - no splitting/combining)<br>
     * - File extensions: no filtering - all files in directory are assumed to be a DataSet<br>
     *
     * @param rootDir Root directory containing the DataSet objects
     */
    public FileDataSetIterator(File rootDir) {
        this(rootDir, true, new Random(), -1, (String[]) null);
    }

    /**
     * Create a FileDataSetIterator with the following default settings:<br>
     * - Recursive: files in subdirectories are included<br>
     * - Randomization: order of examples is randomized with a random RNG seed<br>
     * - Batch size: default (as in the stored DataSets - no splitting/combining)<br>
     * - File extensions: no filtering - all files in directory are assumed to be a DataSet<br>
     *
     * @param rootDirs Root directories containing the DataSet objects. DataSets from all of these directories will
     *                 be included in the iterator output
     */
    public FileDataSetIterator(File... rootDirs) {
        this(rootDirs, true, new Random(), -1, (String[]) null);
    }

    /**
     * Create a FileDataSetIterator with the specified batch size, and the following default settings:<br>
     * - Recursive: files in subdirectories are included<br>
     * - Randomization: order of examples is randomized with a random RNG seed<br>
     * - File extensions: no filtering - all files in directory are assumed to be a DataSet<br>
     *
     * @param rootDir   Root directory containing the saved DataSet objects
     * @param batchSize Batch size. If > 0, DataSets will be split/recombined as required. If <= 0, DataSets will
     *                  simply be loaded and returned unmodified
     */
    public FileDataSetIterator(File rootDir, int batchSize) {
        this(rootDir, batchSize, (String[]) null);
    }

    /**
     * Create a FileDataSetIterator with filtering based on file extensions, and the following default settings:<br>
     * - Recursive: files in subdirectories are included<br>
     * - Randomization: order of examples is randomized with a random RNG seed<br>
     * - Batch size: default (as in the stored DataSets - no splitting/combining)<br>
     *
     * @param rootDir         Root directory containing the saved DataSet objects
     * @param validExtensions May be null. If non-null, only files with one of the specified extensions will be used
     */
    public FileDataSetIterator(File rootDir, String... validExtensions) {
        super(rootDir, -1, validExtensions);
    }

    /**
     * Create a FileDataSetIterator with the specified batch size, filtering based on file extensions, and the
     * following default settings:<br>
     * - Recursive: files in subdirectories are included<br>
     * - Randomization: order of examples is randomized with a random RNG seed<br>
     *
     * @param rootDir         Root directory containing the saved DataSet objects
     * @param batchSize       Batch size. If > 0, DataSets will be split/recombined as required. If <= 0, DataSets will
     *                        simply be loaded and returned unmodified
     * @param validExtensions May be null. If non-null, only files with one of the specified extensions will be used
     */
    public FileDataSetIterator(File rootDir, int batchSize, String... validExtensions) {
        super(rootDir, batchSize, validExtensions);
    }

    /**
     * Create a FileDataSetIterator with all settings specified
     *
     * @param rootDir         Root directory containing the saved DataSet objects
     * @param recursive       If true: include files in subdirectories
     * @param rng             May be null. If non-null, use this to randomize order
     * @param batchSize       Batch size. If > 0, DataSets will be split/recombined as required. If <= 0, DataSets will
     *                        simply be loaded and returned unmodified
     * @param validExtensions May be null. If non-null, only files with one of the specified extensions will be used
     */
    public FileDataSetIterator(File rootDir, boolean recursive, Random rng, int batchSize, String... validExtensions) {
        this(new File[]{rootDir}, recursive, rng, batchSize, validExtensions);
    }

    /**
     * Create a FileDataSetIterator with all settings specified
     *
     * @param rootDirs        Root directories containing the DataSet objects. DataSets from all of these directories will
     *                        be included in the iterator output
     * @param recursive       If true: include files in subdirectories
     * @param rng             May be null. If non-null, use this to randomize order
     * @param batchSize       Batch size. If > 0, DataSets will be split/recombined as required. If <= 0, DataSets will
     *                        simply be loaded and returned unmodified
     * @param validExtensions May be null. If non-null, only files with one of the specified extensions will be used
     */
    public FileDataSetIterator(File[] rootDirs, boolean recursive, Random rng, int batchSize, String... validExtensions) {
        super(rootDirs, recursive, rng, batchSize, validExtensions);
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
    protected void applyPreprocessor(DataSet toPreProcess) {
        if (preProcessor != null) {
            preProcessor.preProcess(toPreProcess);
        }
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
}
