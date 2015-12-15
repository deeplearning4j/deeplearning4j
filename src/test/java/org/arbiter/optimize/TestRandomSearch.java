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
import org.arbiter.optimize.randomsearch.RandomSearchGenerator;
import org.arbiter.optimize.runner.OptimizationRunner;
import org.junit.Test;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;

/**Test random search on the Branin Function:
 * http://www.sfu.ca/~ssurjano/branin.html
 */
public class TestRandomSearch {

    @Test
    public void test(){

        //Define hyperparameter space:


        //Define configuration:

        CandidateGenerator<BraninConfig> candidateGenerator = new RandomSearchGenerator<>(new BraninSpace());
        OptimizationConfiguration<BraninConfig, BraninConfig, Void> configuration =
                new OptimizationConfiguration.Builder<BraninConfig, BraninConfig, Void >()
                .candidateGenerator(candidateGenerator)
                .scoreFunction(new BraninScoreFunction())
                .terminationConditions(new MaxCandidatesCondition(50))
                .maxConcurrentJobs(4)
                .build();

        CandidateExecutor<BraninConfig, BraninConfig, Void> executor =
                new LocalCandidateExecutor<BraninConfig, BraninConfig, Void>(new BraninTaskCreator());

        OptimizationRunner<BraninConfig, BraninConfig, Void> runner
                = new OptimizationRunner<>(configuration, executor);

        runner.execute();


        System.out.println("----- Complete -----");
    }

    private static class BraninSpace implements ModelParameterSpace<BraninConfig>{

        private Random r = new Random(12345);
        @Override
        public BraninConfig randomCandidate() {
            return new BraninConfig(15.0*r.nextDouble()-5, 15*r.nextDouble());    //-5 to +10 and 0 to 15
        }
    }

    @AllArgsConstructor @Data
    private static class BraninConfig {
        private double x1;
        private double x2;
    }

    private static class BraninScoreFunction implements ScoreFunction<BraninConfig,Void>{
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

    private static class BraninTaskCreator implements TaskCreator<BraninConfig,BraninConfig,Void>{
        @Override
        public Callable<OptimizationResult<BraninConfig, BraninConfig>>
            create(final Candidate<BraninConfig> candidate, DataProvider<Void> dataProvider, final ScoreFunction<BraninConfig,Void> scoreFunction) {
            return new Callable<OptimizationResult<BraninConfig, BraninConfig>>() {
                @Override
                public OptimizationResult<BraninConfig, BraninConfig> call() throws Exception {

                    double score = scoreFunction.score(candidate.getValue(),null,null);
                    System.out.println(candidate.getValue().getX1() + "\t" + candidate.getValue().getX2() + "\t" + score);
                    return new OptimizationResult<>(candidate,candidate.getValue(), score, 0);
                }
            };
        }
    }


}
