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

package org.deeplearning4j.arbiter.optimize.api.score;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * ScoreFunction defines the objective of hyperparameter optimization.
 * Specifically, it is used to calculate a score for a given model, relative to the data set provided
 * in the configuration.
 *
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface ScoreFunction extends Serializable {

    /**
     * Calculate and return the score, for the given model and data provider
     *
     * @param model          Model to score
     * @param dataProvider   Data provider - data to use
     * @param dataParameters Parameters for data
     * @return Calculated score
     */
    double score(Object model, DataProvider dataProvider, Map<String, Object> dataParameters);

    /**
     * Calculate and return the score, for the given model and data provider
     *
     * @param model                Model to score
     * @param dataSource           Data source
     * @param dataSourceProperties data source properties
     * @return Calculated score
     */
    double score(Object model, Class<? extends DataSource> dataSource, Properties dataSourceProperties);

    /**
     * Should this score function be minimized or maximized?
     *
     * @return true if score should be minimized, false if score should be maximized
     */
    boolean minimize();

    /**
     * @return The model types supported by this class
     */
    List<Class<?>> getSupportedModelTypes();

    /**
     * @return The data types supported by this class
     */
    List<Class<?>> getSupportedDataTypes();
}
