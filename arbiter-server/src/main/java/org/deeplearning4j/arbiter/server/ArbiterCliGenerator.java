package org.deeplearning4j.arbiter.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.generator.GridSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.ScoreFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Generate an {@link OptimizationConfiguration}
 * via the command line interface.
 * You can then use this configuration json file from
 * {@link ArbiterCliRunner}
 *
 * @author Adam Gibson
 */
public class ArbiterCliGenerator {
    @Parameter(names = {"--searchSpacePath"})
    private String searchSpacePath = null;
    @Parameter(names = {"--candidateType"},required = true)
    private String candidateType = null;
    @Parameter(names = {"--discretizationCount"})
    private int discretizationCount = 5;
    @Parameter(names = {"--gridSearchOrder"})
    private String gridSearchOrder = null;
    @Parameter(names = {"--neuralNetType"},required = true)
    private String neuralNetType = null;
    @Parameter(names = {"--dataSetIteratorClass"},required = true)
    private String dataSetIteratorClass = null;
    @Parameter(names = {"--modelOutputPath"},required = true)
    private String modelOutputPath = null;
    @Parameter(names = {"--score"},required = true)
    private String score = null;
    @Parameter(names = {"--problemType"},required = true)
    private String problemType = CLASSIFICIATION;
    @Parameter(names = {"--configSavePath"},required = true)
    private String configSavePath = null;

    @Parameter(names = {"--duration"},description = "The number of minutes to run for. Default is -1 which means run till convergence.")
    private long duration = -1;
    @Parameter(names = {"--numCandidates"},description = "The number of candidates to generate. Default is 1.")
    private int numCandidates = 1;

    public final static String REGRESSION_MULTI = "regression";
    public final static String REGRESSION = "regression";
    public final static String CLASSIFICIATION = "classification";

    public final static String RANDOM_CANDIDATE = "random";
    public final static String GRID_SEARCH_CANDIDATE = "gridsearch";

    public final static String SEQUENTIAL_ORDER = "sequence";
    public final static String RANDOM_ORDER = "random";

    public final static String COMP_GRAPH = "compgraph";
    public final static String MULTI_LAYER = "multilayer";

    public final static String ACCURACY = "accuracy";
    public final static String F1 = "f1";

    public final static String ACCURACY_MULTI = "accuracy_multi";
    public final static String F1_MULTI = "f1_multi";


    public final static String REGRESSION_SCORE = "regression_score";
    public final static String REGRESSION_SCORE_MULTI = "regression_score_multi";

    public void runMain(String...args) throws Exception  {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch(ParameterException e) {
            System.err.println(e.getMessage());
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            System.exit(1);
        }


        DataProvider dataProvider = new DataSetIteratorFactoryProvider();
        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,dataSetIteratorClass);


        if(neuralNetType.equals(MULTI_LAYER)) {
            MultiLayerSpace multiLayerSpace = loadMultiLayer();
            CandidateGenerator candidateGenerator = null;
            if(candidateType.equals(GRID_SEARCH_CANDIDATE)) {
                candidateGenerator = new RandomSearchGenerator(multiLayerSpace,commands);



            }
            else if(candidateType.equals(RANDOM_CANDIDATE)) {
                candidateGenerator = new RandomSearchGenerator(multiLayerSpace,commands);

            }

            if(problemType.equals(CLASSIFICIATION)) {
                OptimizationConfiguration configuration
                        = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator)
                        .dataProvider(dataProvider)
                        .modelSaver(new FileModelSaver(modelOutputPath))
                        .scoreFunction(scoreFunctionMultiLayerNetwork())
                        .terminationConditions(getConditions())
                        .build();
                FileUtils.writeStringToFile(new File(configSavePath),configuration.toJson());

            }
            else if(problemType.equals(REGRESSION)) {
                OptimizationConfiguration configuration
                        = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator)
                        .dataProvider(dataProvider)
                        .modelSaver(new FileModelSaver(modelOutputPath))
                        .scoreFunction(scoreFunctionMultiLayerNetwork())
                        .terminationConditions(getConditions())
                        .build();
                FileUtils.writeStringToFile(new File(configSavePath),configuration.toJson());

            }


        }
        else if(neuralNetType.equals(COMP_GRAPH)) {
            ComputationGraphSpace computationGraphSpace = loadCompGraph();
            CandidateGenerator candidateGenerator = null;
            if(candidateType.equals(GRID_SEARCH_CANDIDATE)) {
                candidateGenerator = new RandomSearchGenerator(computationGraphSpace,commands);

            }
            else if(candidateType.equals(RANDOM_CANDIDATE)) {
                candidateGenerator = new RandomSearchGenerator(computationGraphSpace,commands);

            }


            if(problemType.equals(CLASSIFICIATION)) {
                OptimizationConfiguration configuration
                        = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator)
                        .dataProvider(dataProvider)
                        .modelSaver(new FileModelSaver(modelOutputPath))
                        .scoreFunction(scoreFunctionCompGraph())
                        .terminationConditions(getConditions())
                        .build();

                FileUtils.writeStringToFile(new File(configSavePath),configuration.toJson());
            }
            else {
                OptimizationConfiguration configuration
                        = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator)
                        .dataProvider(dataProvider)
                        .modelSaver(new FileModelSaver(modelOutputPath))
                        .scoreFunction(scoreFunctionCompGraph())
                        .terminationConditions(getConditions())
                        .build();
                FileUtils.writeStringToFile(new File(configSavePath),configuration.toJson());


            }


        }


    }

    public static void main(String...args) throws Exception {
        new ArbiterCliGenerator().runMain(args);
    }

    private List<TerminationCondition> getConditions() {
        List<TerminationCondition> ret = new ArrayList<>();
        if(duration > 0) {
            ret.add(new MaxTimeCondition(duration,TimeUnit.MINUTES));
        }

        if(numCandidates > 0)
            ret.add(new MaxCandidatesCondition(numCandidates));
        if(ret.isEmpty()) {
            ret.add(new MaxCandidatesCondition(1));
        }
        return ret;
    }


    private GridSearchCandidateGenerator.Mode getMode() {
        if(gridSearchOrder.equals(RANDOM_ORDER))
            return GridSearchCandidateGenerator.Mode.RandomOrder;
        else if(gridSearchOrder.equals(SEQUENTIAL_ORDER)) {
            return GridSearchCandidateGenerator.Mode.Sequential;
        }
        else throw new IllegalArgumentException("Illegal mode " + gridSearchOrder);
    }

    private  ScoreFunction scoreFunctionCompGraph() {
        if(problemType.equals(CLASSIFICIATION)) {
            switch(score) {
                case ACCURACY: return ScoreFunctions.testSetAccuracy();
                case F1: return ScoreFunctions.testSetF1();
                case F1_MULTI : return ScoreFunctions.testSetF1();
                case ACCURACY_MULTI: return ScoreFunctions.testSetAccuracy();

                default: throw new IllegalArgumentException("Score " + score + " not valid for type " + problemType);
            }
        }
        else if(problemType.equals(REGRESSION)) {
            switch(score) {
                case REGRESSION_SCORE: return ScoreFunctions.testSetRegression(RegressionValue.valueOf(score));
                case REGRESSION_SCORE_MULTI: return ScoreFunctions.testSetRegression(RegressionValue.valueOf(score));
                default: throw new IllegalArgumentException("Score " + score + " not valid for type " + problemType);
            }
        }
        throw new IllegalStateException("Illegal problem type " + problemType);
    }

    private  ScoreFunction scoreFunctionMultiLayerNetwork() {
        if(problemType.equals(CLASSIFICIATION)) {
            switch(score) {
                case ACCURACY: return ScoreFunctions.testSetAccuracy();
                case F1: return ScoreFunctions.testSetF1();

                default: throw new IllegalArgumentException("Score " + score + " not valid for type " + problemType);
            }
        }
        else if(problemType.equals(REGRESSION)) {
            switch(score) {
                case REGRESSION_SCORE: return ScoreFunctions.testSetRegression(RegressionValue.valueOf(score));
                default: throw new IllegalArgumentException("Score " + score + " not valid for type " + problemType);

            }
        }
        throw new IllegalStateException("Illegal problem type " + problemType);
    }

    private ComputationGraphSpace loadCompGraph() throws Exception {
        ComputationGraphSpace multiLayerSpace = ComputationGraphSpace.fromJson(FileUtils.readFileToString(new File(searchSpacePath)));
        return multiLayerSpace;
    }

    private MultiLayerSpace loadMultiLayer() throws Exception {
        MultiLayerSpace multiLayerSpace = MultiLayerSpace.fromJson(FileUtils.readFileToString(new File(searchSpacePath)));
        return multiLayerSpace;
    }
}
