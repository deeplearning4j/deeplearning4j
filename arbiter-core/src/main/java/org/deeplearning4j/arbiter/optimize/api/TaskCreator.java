/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.api;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;

import java.util.concurrent.Callable;

/**
 * The TaskCreator is used to take a candidate configuration, data provider and score function, and create something
 * that can be executed as a Callable
 *
 * @param <C> Type of the candidate configuration
 * @param <M> Type of the learned models
 * @param <D> Type of data used to train models
 * @param <A> Type of any additional evaluation
 * @author Alex Black
 */
public interface TaskCreator<C, M, D, A> {

    /**
     * Generate a callable that can be executed to conduct the training of this model (given the model configuration)
     *
     * @param candidate      Candidate (model) configuration to be trained
     * @param dataProvider   DataProvider, for the data
     * @param scoreFunction  Score function to be used to evaluate the model
     * @param statusListener Status listener, that can be used to provide status updates to the UI, as the task executes
     * @return A callable that returns an OptimizationResult, once optimization is complete
     */
    Callable<OptimizationResult<C, M, A>> create(Candidate<C> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction,
                                                 UICandidateStatusListener statusListener);
}
