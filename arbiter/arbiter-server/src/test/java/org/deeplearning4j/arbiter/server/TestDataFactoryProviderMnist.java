package org.deeplearning4j.arbiter.server;

import lombok.AllArgsConstructor;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

@AllArgsConstructor
public class TestDataFactoryProviderMnist implements DataSetIteratorFactory {

    private int batchSize;
    private int terminationIter;

    public TestDataFactoryProviderMnist(){
        this(16, 10);
    }

    @Override
    public DataSetIterator create() {
        try {
            return new EarlyTerminationDataSetIterator(new MnistDataSetIterator(batchSize, true, 12345), terminationIter);
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }
}
