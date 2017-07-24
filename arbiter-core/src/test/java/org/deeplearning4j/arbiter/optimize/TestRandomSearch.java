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
package org.deeplearning4j.arbiter.optimize;

import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.impl.LoggingStatusListener;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * Test random search on the Branin Function:
 * http://www.sfu.ca/~ssurjano/branin.html
 */
public class TestRandomSearch {

    @Test
    public void test() throws Exception {
        Map<String, Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, new HashMap<>());

        //Define configuration:
        CandidateGenerator candidateGenerator = new RandomSearchGenerator(new TestGridSearch.BraninSpace(), commands);
        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator).scoreFunction(new TestGridSearch.BraninScoreFunction())
                        .terminationConditions(new MaxCandidatesCondition(50)).build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new TestGridSearch.BraninTaskCreator());

        runner.addListeners(new LoggingStatusListener());
        runner.execute();


        System.out.println("----- Complete -----");
    }


}
