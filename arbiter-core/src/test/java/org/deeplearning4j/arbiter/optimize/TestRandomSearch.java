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

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.Status;
import org.deeplearning4j.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;
import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;
import org.deeplearning4j.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.arbiter.util.WebUtils;
import org.deeplearning4j.arbiter.optimize.api.*;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.Callable;

/**
 *
 * Test random search on the Branin Function:
 * http://www.sfu.ca/~ssurjano/branin.html
 */
public class TestRandomSearch {
    @Test
    @Ignore
    public void test() throws Exception {
        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,new HashMap<>());

        //Define configuration:
        CandidateGenerator<BraninConfig> candidateGenerator = new RandomSearchGenerator<>(new BraninSpace(),commands);
        OptimizationConfiguration<BraninConfig, BraninConfig, Void, Void> configuration =
                new OptimizationConfiguration.Builder<BraninConfig, BraninConfig, Void, Void >()
                        .candidateGenerator(candidateGenerator)
                        .scoreFunction(new BraninScoreFunction())
                        .terminationConditions(new MaxCandidatesCondition(50))
                        .build();

        IOptimizationRunner<BraninConfig, BraninConfig, Void> runner
                = new LocalOptimizationRunner<>(configuration, new BraninTaskCreator());
//        runner.addListeners(new LoggingOptimizationRunnerStatusListener());

        ArbiterUIServer server = ArbiterUIServer.getInstance();
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));
        runner.execute();


        System.out.println("----- Complete -----");
    }

    public static class BraninSpace implements ParameterSpace<BraninConfig> {
        private int[] indices;
        private ParameterSpace<Double> first = new ContinuousParameterSpace(-5,10);
        private ParameterSpace<Double> second = new ContinuousParameterSpace(0,15);

        @Override
        public BraninConfig getValue(double[] parameterValues) {
            double f = first.getValue(parameterValues);
            double s = second.getValue(parameterValues);
            return new BraninConfig(f,s);   //-5 to +10 and 0 to 15
        }

        @Override
        public int numParameters() {
            return 2;
        }

        @Override
        public List<ParameterSpace> collectLeaves() {
            List<ParameterSpace> list = new ArrayList<>();
            list.addAll(first.collectLeaves());
            list.addAll(second.collectLeaves());
            return list;
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        @Override
        public void setIndices(int... indices) {
            throw new UnsupportedOperationException();
        }
    }

    @AllArgsConstructor @Data
    public static class BraninConfig {
        private double x1;
        private double x2;
    }

    public static class BraninScoreFunction implements ScoreFunction<BraninConfig,Void> {
        private static final double a = 1.0;
        private static final double b = 5.1 / (4.0 * Math.PI * Math.PI );
        private static final double c = 5.0 / Math.PI;
        private static final double r = 6.0;
        private static final double s = 10.0;
        private static final double t = 1.0 / (8.0 * Math.PI);

        @Override
        public double score(BraninConfig model, DataProvider<Void> data, Map<String,Object> dataParameters) {
            double x1 = model.getX1();
            double x2 = model.getX2();

            return a*Math.pow(x2 - b*x1*x1 + c*x1 - r,2.0) + s*(1-t)*Math.cos(x1) + s;
        }

        @Override
        public boolean minimize() {
            return true;
        }
    }

    public static class BraninTaskCreator implements TaskCreator<BraninConfig,BraninConfig,Void,Void> {
        @Override
        public Callable<OptimizationResult<BraninConfig, BraninConfig,Void>> create(final Candidate<BraninConfig> candidate,
                                                                                    DataProvider<Void> dataProvider, final ScoreFunction<BraninConfig,Void> scoreFunction,
                                                                                    final UICandidateStatusListener statusListener) {

            if(statusListener != null){
                statusListener.reportStatus(Status.Created,new ComponentText("Config: " + candidate.toString(), null));
            }

            return new Callable<OptimizationResult<BraninConfig, BraninConfig, Void>>() {
                @Override
                public OptimizationResult<BraninConfig, BraninConfig, Void> call() throws Exception {

                    if(statusListener != null) {
                        statusListener.reportStatus(Status.Running,
                                new ComponentText("Config: " + candidate.toString(), null)
                        );
                    }

                    double score = scoreFunction.score(candidate.getValue(),null,null);
                    System.out.println(candidate.getValue().getX1() + "\t" + candidate.getValue().getX2() + "\t" + score);

                    Thread.sleep(500);
                    if(statusListener != null) {
                        statusListener.reportStatus(Status.Complete,
                                new ComponentText("Config: " + candidate.toString(), null),
                                new ComponentText("Score: " + score, null)
                        );
                    }

                    return new OptimizationResult<>(candidate,candidate.getValue(), score, candidate.getIndex(), null);
                }
            };
        }
    }


}
