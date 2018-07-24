package org.deeplearning4j.arbiter.optimize.api.data;

import java.io.Serializable;
import java.util.Properties;

/**
 * DataSource: defines where the data should come from for training and testing.
 * Note that implementations must have a no-argument contsructor
 *
 * @author Alex Black
 */
public interface DataSource extends Serializable {

    /**
     * Configure the current data source with the specified properties
     * Note: These properties are fixed for the training instance, and are optionally provided by the user
     * at the configuration stage.
     * The properties could be anything - and are usually specific to each DataSource implementation.
     * For example, values such as batch size could be set using these properties
     * @param properties Properties to apply to the data source instance
     */
    void configure(Properties properties);

    /**
     * Get test data to be used for the optimization. Usually a DataSetIterator or MultiDataSetIterator
     */
    Object trainData();

    /**
     * Get test data to be used for the optimization. Usually a DataSetIterator or MultiDataSetIterator
     */
    Object testData();

    /**
     * The type of data returned by {@link #trainData()} and {@link #testData()}.
     * Usually DataSetIterator or MultiDataSetIterator
     * @return Class of the objects returned by trainData and testData
     */
    Class<?> getDataType();

}
