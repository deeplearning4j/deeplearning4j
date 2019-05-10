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

package org.deeplearning4j.arbiter.optimize.api;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;

import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;

/**
 * The TaskCreator is used to take a candidate configuration, data provider and score function, and create something
 * that can be executed as a Callable
 *
 * @author Alex Black
 */
public interface TaskCreator {

    /**
     * Generate a callable that can be executed to conduct the training of this model (given the model configuration)
     *
     * @param candidate       Candidate (model) configuration to be trained
     * @param dataProvider    DataProvider, for the data
     * @param scoreFunction   Score function to be used to evaluate the model
     * @param statusListeners Status listeners, that can be used for callbacks (to UI, for example)
     * @return A callable that returns an OptimizationResult, once optimization is complete
     */
    @Deprecated
    Callable<OptimizationResult> create(Candidate candidate, DataProvider dataProvider, ScoreFunction scoreFunction,
                                        List<StatusListener> statusListeners, IOptimizationRunner runner);

    /**
     * Generate a callable that can be executed to conduct the training of this model (given the model configuration)
     *
     * @param candidate            Candidate (model) configuration to be trained
     * @param dataSource           Data source
     * @param dataSourceProperties Properties (may be null) for the data source
     * @param scoreFunction        Score function to be used to evaluate the model
     * @param statusListeners      Status listeners, that can be used for callbacks (to UI, for example)
     * @return A callable that returns an OptimizationResult, once optimization is complete
     */
    Callable<OptimizationResult> create(Candidate candidate, Class<? extends DataSource> dataSource, Properties dataSourceProperties,
                                        ScoreFunction scoreFunction, List<StatusListener> statusListeners, IOptimizationRunner runner);
}
