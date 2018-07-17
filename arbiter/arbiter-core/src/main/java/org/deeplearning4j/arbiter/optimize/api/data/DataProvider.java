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

import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.Map;

/**
 * DataProvider interface abstracts out the providing of data
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface DataProvider extends Serializable {

    /**
     * Get training data given some parameters for the data.
     * Data parameters map is used to specify things like batch
     * size data preprocessing
     *
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    Object trainData(Map<String, Object> dataParameters);

    /**
     * Get training data given some parameters for the data. Data parameters map is used to specify things like batch
     * size data preprocessing
     *
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    Object testData(Map<String, Object> dataParameters);

    Class<?> getDataType();
}
