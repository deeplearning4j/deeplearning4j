package org.deeplearning4j.datasets.iterator.parallel;

import lombok.AllArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.ParallelDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class JointParallelDataSetIterator {
    protected List<DataSetIterator> asyncIterators = new ArrayList<>();
    protected boolean enforceSingleDevice;
    protected int bufferSizePerDevice;
    protected int numProducers;

    protected AtomicLong counter = new AtomicLong(0);

    public JointParallelDataSetIterator(List<DataSetIterator> iterators, boolean singleDeviceMode, int bufferSize) {
        this.enforceSingleDevice = singleDeviceMode;
        this.bufferSizePerDevice = bufferSize;
        this.numProducers = iterators.size();

        if (numProducers == 0)
            throw new DL4JInvalidInputException("You can't start ParallelDataSetIterator without input data");

        initializeIterators(iterators);
    }

    protected void initializeIterators(List<DataSetIterator> originals) {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        if (originals.size() % numDevices != 0)
            log.error("WARNING: number of splits doesn't match number of devices!");

        int cnt = 0;
        for (DataSetIterator iterator: originals) {
            int cDev = cnt % numDevices;
            asyncIterators.add(new AsyncDataSetIterator(iterator, bufferSizePerDevice, true, cDev));
            cnt++;
        }
    }

    public boolean hasNext() {
        // TODO: configurable probably?
        return true;
    }

    public DataSet next() {
        return asyncIterators.get((int)(counter.getAndIncrement() % numProducers)).next();
    }


    public static class Builder {
        private List<DataSetIterator> iterators = new ArrayList<>();
        private boolean enforceSingleDevice = true;
        private int bufferSize = 4;

        public Builder addSourceIterator(@NonNull DataSetIterator iterator) {
            if (!iterator.asyncSupported())
                throw new DL4JInvalidInputException("Source iterators should support async mode");

            //TODO: add strict equality check here, we don't want it equal
            iterators.add(iterator);
            return this;
        }

        public Builder setBufferSizePerSplit(int bufferSize) {
            this.bufferSize = bufferSize;
            return this;
        }


        public Builder enforceSingleDevice(boolean reallyEnforce) {
            this.enforceSingleDevice = reallyEnforce;
            return this;
        }


        public JointParallelDataSetIterator build() {
            JointParallelDataSetIterator jpdsi = new JointParallelDataSetIterator(iterators, enforceSingleDevice, bufferSize);

            return jpdsi;
        }
    }
}
