package org.deeplearning4j.datasets.iterator.parallel;

import com.google.common.collect.Lists;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.FileSplitDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.FileCallback;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class FileSplitParallelDataSetIterator extends BaseParallelDataSetIterator {

    public static final String DEFAULT_PATTERN = "dataset-%d.bin";
    private String pattern;
    private int buffer;

    protected List<DataSetIterator> asyncIterators = new ArrayList<>();

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull String pattern,
                    @NonNull FileCallback callback) {
        this(rootFolder, pattern, callback, Nd4j.getAffinityManager().getNumberOfDevices());
    }

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull String pattern,
                    @NonNull FileCallback callback, int numThreads) {
        this(rootFolder, pattern, callback, numThreads, InequalityHandling.STOP_EVERYONE);
    }

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull String pattern,
                    @NonNull FileCallback callback, int numThreads, @NonNull InequalityHandling inequalityHandling) {
        this(rootFolder, pattern, callback, numThreads, 2, inequalityHandling);
    }

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull String pattern,
                    @NonNull FileCallback callback, int numThreads, int bufferPerThread,
                    @NonNull InequalityHandling inequalityHandling) {
        super(numThreads);

        if (!rootFolder.exists() || !rootFolder.isDirectory())
            throw new IllegalArgumentException("Root folder should point to existing folder");

        this.pattern = pattern;
        this.inequalityHandling = inequalityHandling;
        this.buffer = bufferPerThread;

        String modifiedPattern = pattern.replaceAll("\\%d", ".*.");

        IOFileFilter fileFilter = new RegexFileFilter(modifiedPattern);


        List<File> files = new ArrayList<>(FileUtils.listFiles(rootFolder, fileFilter, null));
        log.debug("Files found: {}; Producers: {}", files.size(), numProducers);

        if (files.isEmpty())
            throw new IllegalArgumentException("No suitable files were found");

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        int cnt = 0;
        for (List<File> part : Lists.partition(files, files.size() / numThreads)) {
            // discard remainder
            if (cnt >= numThreads)
                break;

            int cDev = cnt % numDevices;
            asyncIterators.add(new AsyncDataSetIterator(new FileSplitDataSetIterator(part, callback), bufferPerThread,
                            true, cDev));
            cnt++;
        }

    }

    @Override
    public boolean hasNextFor(int consumer) {
        if (consumer >= numProducers || consumer < 0)
            throw new ND4JIllegalStateException("Non-existent consumer was requested");

        return asyncIterators.get(consumer).hasNext();
    }

    @Override
    public DataSet nextFor(int consumer) {
        if (consumer >= numProducers || consumer < 0)
            throw new ND4JIllegalStateException("Non-existent consumer was requested");

        return asyncIterators.get(consumer).next();
    }

    @Override
    protected void reset(int consumer) {
        if (consumer >= numProducers || consumer < 0)
            throw new ND4JIllegalStateException("Non-existent consumer was requested");

        asyncIterators.get(consumer).reset();
    }



}
