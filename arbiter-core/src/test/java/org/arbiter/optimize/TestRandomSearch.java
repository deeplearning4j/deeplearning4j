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
package org.arbiter.optimize;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.arbiter.optimize.api.*;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.executor.CandidateExecutor;
import org.arbiter.optimize.executor.local.LocalCandidateExecutor;
import org.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.arbiter.optimize.runner.OptimizationRunner;
import org.arbiter.optimize.runner.Status;
import org.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;
import org.arbiter.optimize.ui.ArbiterUIServer;
import org.arbiter.optimize.ui.components.RenderableComponentString;
import org.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.arbiter.util.WebUtils;
import org.canova.api.util.ClassPathResource;
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
    public static Logger log = LoggerFactory.getLogger(TestRandomSearch.class);
    @Test
    @Ignore
    public void test() throws Exception {

        //Define configuration:
        CandidateGenerator<BraninConfig> candidateGenerator = new RandomSearchGenerator<>(new BraninSpace());
        OptimizationConfiguration<BraninConfig, BraninConfig, Void, Void> configuration =
                new OptimizationConfiguration.Builder<BraninConfig, BraninConfig, Void, Void >()
                        .candidateGenerator(candidateGenerator)
                        .scoreFunction(new BraninScoreFunction())
                        .terminationConditions(new MaxCandidatesCondition(50))
                        .build();

        CandidateExecutor<BraninConfig, BraninConfig, Void, Void> executor =
                new LocalCandidateExecutor<>(new BraninTaskCreator());

        OptimizationRunner<BraninConfig, BraninConfig, Void, Void> runner
                = new OptimizationRunner<>(configuration, executor);
//        runner.addListeners(new LoggingOptimizationRunnerStatusListener());

        ArbiterUIServer server = new ArbiterUIServer();
        String[] str = new String[]{"server", new ClassPathResource("dropwizard.yml").getFile().getAbsolutePath()};
        server.run(str);
        WebUtils.tryOpenBrowser("http://localhost:8080/arbiter", log);    //TODO don't hardcode
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));
        runner.execute();


        System.out.println("----- Complete -----");
    }

    public static class BraninSpace implements ParameterSpace<BraninConfig>{
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

    public static class BraninScoreFunction implements ScoreFunction<BraninConfig,Void>{
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
    }

    public static class BraninTaskCreator implements TaskCreator<BraninConfig,BraninConfig,Void,Void>{
        @Override
        public Callable<OptimizationResult<BraninConfig, BraninConfig,Void>> create(final Candidate<BraninConfig> candidate,
                                                                                    DataProvider<Void> dataProvider, final ScoreFunction<BraninConfig,Void> scoreFunction,
                                                                                    final UICandidateStatusListener statusListener) {

            if(statusListener != null){
                statusListener.reportStatus(Status.Created,new RenderableComponentString("Config: " + candidate.toString()));
            }

            return new Callable<OptimizationResult<BraninConfig, BraninConfig, Void>>() {
                @Override
                public OptimizationResult<BraninConfig, BraninConfig, Void> call() throws Exception {

                    if(statusListener != null) {
                        statusListener.reportStatus(Status.Running,
                                new RenderableComponentString("Config: " + candidate.toString())
                        );
                    }

                    double score = scoreFunction.score(candidate.getValue(),null,null);
                    System.out.println(candidate.getValue().getX1() + "\t" + candidate.getValue().getX2() + "\t" + score);

                    Thread.sleep(500);
                    if(statusListener != null) {
                        statusListener.reportStatus(Status.Complete,
                                new RenderableComponentString("Config: " + candidate.toString()),
                                new RenderableComponentString("Score: " + score)
                        );
                    }

                    return new OptimizationResult<>(candidate,candidate.getValue(), score, candidate.getIndex(), null);
                }
            };
        }
    }


}
