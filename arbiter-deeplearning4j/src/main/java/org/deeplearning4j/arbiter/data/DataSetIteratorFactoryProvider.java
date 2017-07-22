package org.deeplearning4j.arbiter.data;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

import java.util.Map;

/**
 * This is a {@link DataProvider} for
 * an {@link DataSetIteratorFactory} which
 * based on a key of {@link DataSetIteratorFactoryProvider#FACTORY_KEY}
 * will create {@link org.nd4j.linalg.dataset.api.iterator.DataSetIterator}
 * for use with arbiter.
 *
 * This {@link DataProvider} is mainly meant for use for command line driven
 * applications.
 *
 * @author Adam Gibson
 */
public class DataSetIteratorFactoryProvider implements DataProvider {

    public final static String FACTORY_KEY = "org.deeplearning4j.arbiter.data.data.factory";

    /**
     * Get training data given some parameters for the data.
     * Data parameters map is used to specify things like batch
     * size data preprocessing
     *
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    @Override
    public DataSetIteratorFactory trainData(Map<String, Object> dataParameters) {
        return create(dataParameters);
    }

    /**
     * Get training data given some parameters for the data. Data parameters map
     * is used to specify things like batch
     * size data preprocessing
     *
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    @Override
    public DataSetIteratorFactory testData(Map<String, Object> dataParameters) {
        return create(dataParameters);
    }

    @Override
    public Class<?> getDataType() {
        return DataSetIteratorFactory.class;
    }

    private DataSetIteratorFactory create(Map<String, Object> dataParameters) {
        if (!dataParameters.containsKey(FACTORY_KEY))
            throw new IllegalArgumentException(
                            "No data set iterator factory class found. Please specify a class name with key "
                                            + FACTORY_KEY);
        String value = dataParameters.get(FACTORY_KEY).toString();
        try {
            Class<? extends DataSetIteratorFactory> clazz =
                            (Class<? extends DataSetIteratorFactory>) Class.forName(value);
            return clazz.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
