package org.deeplearning4j.datasets.iterator.parallel;


import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class JointParallelDataSetIterator extends BaseParallelDataSetIterator {
    protected List<DataSetIterator> asyncIterators = new ArrayList<>();
    protected boolean enforceSingleDevice;
    protected int bufferSizePerDevice;


    public JointParallelDataSetIterator(@NonNull List<DataSetIterator> iterators, boolean singleDeviceMode,
                    int bufferSize, @NonNull InequalityHandling inequalityHandling) {
        super(iterators.size());
        this.enforceSingleDevice = singleDeviceMode;
        this.bufferSizePerDevice = bufferSize;
        this.numProducers = iterators.size();
        this.inequalityHandling = inequalityHandling;

        if (numProducers == 0)
            throw new IllegalArgumentException("You can't start ParallelDataSetIterator without input data");

        initializeIterators(iterators);
    }

    protected void initializeIterators(List<DataSetIterator> originals) {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        if (originals.size() % numDevices != 0)
            log.error("WARNING: number of splits doesn't match number of devices!");

        int cnt = 0;
        for (DataSetIterator iterator : originals) {
            int cDev = cnt % numDevices;
            asyncIterators.add(new AsyncDataSetIterator(iterator, bufferSizePerDevice, true, cDev));
            cnt++;
        }
    }

    public boolean hasNextFor(int consumer) {
        if (consumer >= numProducers || consumer < 0)
            throw new ND4JIllegalStateException("Non-existent consumer was requested");

        return asyncIterators.get(consumer).hasNext();
    }


    public DataSet nextFor(int consumer) {
        if (consumer >= numProducers || consumer < 0)
            throw new ND4JIllegalStateException("Non-existent consumer was requested");

        return asyncIterators.get(consumer).next();
    }

    protected void reset(int consumer) {
        if (consumer >= numProducers || consumer < 0)
            throw new ND4JIllegalStateException("Non-existent consumer was requested");

        asyncIterators.get(consumer).reset();
    }


    public static class Builder {
        private List<DataSetIterator> iterators = new ArrayList<>();
        private boolean enforceSingleDevice = true;
        private int bufferSize = 4;
        private InequalityHandling inequalityHandling;

        public Builder(@NonNull InequalityHandling inequalityHandling) {
            this.inequalityHandling = inequalityHandling;
        }

        public Builder(@NonNull List<DataSetIterator> iterators, @NonNull InequalityHandling inequalityHandling) {
            this.inequalityHandling = inequalityHandling;

            for (DataSetIterator iterator : iterators)
                addSourceIterator(iterator);
        }


        public Builder addSourceIterator(@NonNull DataSetIterator iterator) {
            if (!iterator.asyncSupported())
                throw new IllegalArgumentException("Source iterators should support async mode");

            //TODO: add strict equality check here, we don't want it equal
            if (!hasIterator(iterator))
                iterators.add(iterator);
            else
                throw new IllegalArgumentException("You can't put equal iterators into this joint iterator");

            return this;
        }

        protected boolean hasIterator(DataSetIterator iterator) {
            for (DataSetIterator iter : iterators) {
                if (iter == iterator)
                    return true;
            }

            return false;
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
            JointParallelDataSetIterator jpdsi = new JointParallelDataSetIterator(iterators, enforceSingleDevice,
                            bufferSize, inequalityHandling);

            return jpdsi;
        }
    }
}
