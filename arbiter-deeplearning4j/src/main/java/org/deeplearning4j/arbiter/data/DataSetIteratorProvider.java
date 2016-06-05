package org.deeplearning4j.arbiter.data;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Created by Alex on 5/06/2016.
 */
public class DataSetIteratorProvider implements DataProvider<DataSetIterator> {

    private final DataSetIterator trainData;
    private final DataSetIterator testData;
    private final boolean resetBeforeReturn;

    public DataSetIteratorProvider(DataSetIterator trainData, DataSetIterator testData) {
        this(trainData, testData, true);
    }

    public DataSetIteratorProvider(DataSetIterator trainData, DataSetIterator testData, boolean resetBeforeReturn){
        this.trainData = trainData;
        this.testData = testData;
        this.resetBeforeReturn = resetBeforeReturn;
    }

    @Override
    public DataSetIterator trainData(Map<String, Object> dataParameters) {
        if(resetBeforeReturn) trainData.reset();    //Same iterator might be used multiple times by different models
        return trainData;
    }

    @Override
    public DataSetIterator testData(Map<String, Object> dataParameters) {
        if(resetBeforeReturn) testData.reset();
        return testData;
    }
}
