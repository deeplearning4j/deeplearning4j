package org.deeplearning4j.integration.util;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

@Data
public class CountingMultiDataSetIterator implements MultiDataSetIterator {

    private MultiDataSetIterator underlying;
    private int currIter = 0;
    private IntArrayList iterAtReset = new IntArrayList();
    private boolean tbptt;
    private int tbpttLength;

    public CountingMultiDataSetIterator(MultiDataSetIterator underlying, boolean tbptt, int tbpttLength){
        this.underlying = underlying;
        this.tbptt = tbptt;
        this.tbpttLength = tbpttLength;
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
        MultiDataSet mds = underlying.next();
        if(tbptt){
            INDArray f = mds.getFeatures(0);
            if(f.rank() == 3){
                int numSegments = (int)Math.ceil(f.size(2) / (double)tbpttLength);
                currIter += numSegments;
            }
        } else {
            currIter++;
        }
        return mds;
    }
}
