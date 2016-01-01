package org.arbiter.optimize.api.data;

import java.util.Map;

public interface DataProvider<D> {

    /** Get training data given some parameters for the data. Data parameters map is used to specify things like batch
     * size data preprocessing
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    D trainData(Map<String, Object> dataParameters);

    /** Get training data given some parameters for the data. Data parameters map is used to specify things like batch
     * size data preprocessing
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    D testData(Map<String, Object> dataParameters);

}
