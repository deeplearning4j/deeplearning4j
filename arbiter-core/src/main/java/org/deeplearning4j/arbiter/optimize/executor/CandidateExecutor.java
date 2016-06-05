/*
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
package org.deeplearning4j.arbiter.optimize.executor;

import com.google.common.util.concurrent.ListenableFuture;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;

import java.util.List;

public interface CandidateExecutor<C,M,D,A> {

    ListenableFuture<OptimizationResult<C,M,A>> execute(Candidate<C> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction);

    List<ListenableFuture<OptimizationResult<C,M,A>>> execute(List<Candidate<C>> candidates, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction);

    int maxConcurrentTasks();

    /** Shut down the executor, and immediately cancel all remaining tasks */
    void shutdown();

}
