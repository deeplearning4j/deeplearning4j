package integration.util;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import lombok.Data;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

@Data
public class CountingMultiDataSetIterator implements MultiDataSetIterator {

    private MultiDataSetIterator underlying;
    private int currIter = 0;
    private IntArrayList iterAtReset = new IntArrayList();

    public CountingMultiDataSetIterator(MultiDataSetIterator underlying){
        this.underlying = underlying;
    }

    @Override
    public MultiDataSet next(int i) {
        currIter++;
        return underlying.next(i);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        underlying.setPreProcessor(multiDataSetPreProcessor);
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return underlying.getPreProcessor();
    }

    @Override
    public boolean resetSupported() {
        return underlying.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return underlying.asyncSupported();
    }

    @Override
    public void reset() {
        underlying.reset();
        iterAtReset.add(currIter);
        currIter = 0;
    }

    @Override
    public boolean hasNext() {
        return underlying.hasNext();
    }

    @Override
    public MultiDataSet next() {
        currIter++;
        return underlying.next();
    }
}
