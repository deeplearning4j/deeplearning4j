package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.cache.DataSetCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Created by anton on 7/16/16.
 */
public class CachingDataSetIterator implements DataSetIterator {
    private static final Logger log = LoggerFactory.getLogger(DataSetCache.class);

    private DataSetIterator sourceIterator;
    private DataSetCache cache;
    private DataSetPreProcessor preProcessor;
    private String namespace;
    private int currentIndex = 0;
    private boolean usingCache = false;

    public CachingDataSetIterator(DataSetIterator sourceIterator, DataSetCache cache, String namespace) {
        this.sourceIterator = sourceIterator;
        this.cache = cache;
        this.preProcessor = null;
        this.namespace = namespace;
        this.currentIndex = 0;

        this.usingCache = cache.isComplete(namespace);
    }

    public CachingDataSetIterator(DataSetIterator sourceIterator, DataSetCache cache) {
        this(sourceIterator, cache, "default");
    }

    private String makeKey(int index) {
        return String.format("data-set-cache-%s-%06d.bin", namespace, index);
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalExamples() {
        return sourceIterator.totalExamples();
    }

    @Override
    public int inputColumns() {
        return sourceIterator.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return sourceIterator.totalOutcomes();
    }

    @Override
    public void reset() {
        currentIndex = 0;
    }

    @Override
    public int batch() {
        return sourceIterator.numExamples();
    }

    @Override
    public int cursor() {
        return currentIndex;
    }

    @Override
    public int numExamples() {
        return sourceIterator.numExamples();
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
        return sourceIterator.getLabels();
    }

    @Override
    public boolean hasNext() {
        if (usingCache) {
            return cache.contains(makeKey(currentIndex));
        } else {
            if (sourceIterator.hasNext()) {
                return true;
            } else {
                usingCache = true;
                cache.setComplete(namespace, true);
                return false;
            }
        }
    }

    @Override
    public DataSet next() {
        String key = makeKey(currentIndex);

        DataSet ds;

        if (usingCache) {
            ds = cache.get(key);
        } else {
            ds = sourceIterator.next();
            if (preProcessor != null) {
                preProcessor.preProcess(ds);
            }
            cache.put(key, ds);
        }

        currentIndex += 1;

        return ds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
