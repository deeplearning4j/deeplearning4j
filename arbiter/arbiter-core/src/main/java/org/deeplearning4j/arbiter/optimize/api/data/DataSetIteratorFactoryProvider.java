/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.api.data;

import lombok.Data;
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
@Data
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
        if (dataParameters == null)
            throw new IllegalArgumentException(
                            "Data parameters is null. Please specify a class name to create a dataset iterator.");
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
