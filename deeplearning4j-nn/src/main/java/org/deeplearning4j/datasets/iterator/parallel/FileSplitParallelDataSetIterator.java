package org.deeplearning4j.datasets.iterator.parallel;

import com.google.common.collect.Lists;
import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.NameFileFilter;
import org.deeplearning4j.datasets.iterator.callbacks.FileCallback;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class FileSplitParallelDataSetIterator extends BaseParallelDataSetIterator {

    public static final String DEFAULT_PATTERN = "dataset-%d.bin";
    private String pattern;

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull String pattern, @NonNull FileCallback callback) {
        this(rootFolder, pattern, callback, Nd4j.getAffinityManager().getNumberOfDevices());
    }

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull String pattern, @NonNull FileCallback callback, int numThreads) {
        super(numThreads);

        if (!rootFolder.exists() || !rootFolder.isDirectory())
            throw new DL4JInvalidInputException("Root folder should point to existing folder");

        this.pattern = pattern;

        String modifiedPattern = pattern.replaceAll("%d","*");

        IOFileFilter fileFilter = new NameFileFilter(modifiedPattern);


        List<File> files = new ArrayList<>(FileUtils.listFiles(rootFolder, fileFilter, null));

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (List<File> part: Lists.partition(files, files.size() / numThreads)) {

        }

    }

    @Override
    public boolean hasNextFor(int consumer) {
        return false;
    }

    @Override
    public DataSet nextFor(int consumer) {
        return null;
    }

    @Override
    protected void reset(int consumer) {

    }
}
