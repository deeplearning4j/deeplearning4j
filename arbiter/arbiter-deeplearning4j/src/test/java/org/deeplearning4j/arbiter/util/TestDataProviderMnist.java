package org.deeplearning4j.arbiter.util;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

@AllArgsConstructor
public class TestDataProviderMnist implements DataProvider {

    private int batchSize;
    private int terminationIter;

    public TestDataProviderMnist(){
        this(32, 10);
    }

    @Override
    public Object trainData(Map<String, Object> dataParameters) {
        try {
            return new EarlyTerminationDataSetIterator(new MnistDataSetIterator(batchSize, true, 12345), terminationIter);
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public Object testData(Map<String, Object> dataParameters) {
        try {
            return new EarlyTerminationDataSetIterator(new MnistDataSetIterator(batchSize, false, 12345), terminationIter);
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public Class<?> getDataType() {
        return DataSetIterator.class;
    }


}
