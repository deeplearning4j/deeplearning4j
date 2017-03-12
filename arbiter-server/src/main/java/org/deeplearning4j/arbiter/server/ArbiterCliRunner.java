package org.deeplearning4j.arbiter.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.GraphConfiguration;
import org.deeplearning4j.arbiter.evaluator.graph.GraphClassificationDataSetEvaluator;
import org.deeplearning4j.arbiter.evaluator.graph.GraphRegressionDataSetEvaluator;
import org.deeplearning4j.arbiter.evaluator.multilayer.ClassificationEvaluator;
import org.deeplearning4j.arbiter.evaluator.multilayer.RegressionDataSetEvaluator;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by agibsonccc on 3/12/17.
 */
public class ArbiterCliRunner {
    @Parameter(names = {"--modelSavePath"})
    private String modelSavePath = System.getProperty("java.io.tmpdir");
    @Parameter(names = {"--optimizationConfigPath"})
    private String optimizationConfigPath = null;
    @Parameter(names = {"--problemType"})
    private String problemType = CLASSIFICIATION;
    @Parameter(names = {"--regressionType"})
    private String regressionType = null;
    @Parameter(names = {"--dataSetIteratorClass"},required = true)
    private String dataSetIteratorClass = null;
    @Parameter(names = {"--neuralNetType"},required = true)
    private String neuralNetType = null;

    public final static String CLASSIFICIATION = "classification";
    public final static String REGRESSION = "regression";


    public final static String COMP_GRAPH = "compgraph";
    public final static String MULTI_LAYER_NETWORK = "multilayernetwork";

    public void runMain(String...args) throws Exception {
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

        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,dataSetIteratorClass);

        File f = new File(modelSavePath);

        if(f.exists()) f.delete();
        f.mkdir();
        f.deleteOnExit();

        if(problemType.equals(REGRESSION)) {
            if(neuralNetType.equals(COMP_GRAPH)) {
                OptimizationConfiguration<GraphConfiguration,ComputationGraph,Object,Double> configuration
                        = OptimizationConfiguration.fromJson(
                        FileUtils.readFileToString(new File(optimizationConfigPath)),
                        GraphConfiguration.class,
                        ComputationGraph.class,
                        Object.class,
                        Double.class);

                IOptimizationRunner<GraphConfiguration,ComputationGraph,Double> runner
                        = new LocalOptimizationRunner<>(configuration, new ComputationGraphTaskCreator<>(
                        new GraphRegressionDataSetEvaluator(
                                RegressionValue.valueOf(regressionType),
                                commands)));
                runner.execute();
            }
            else if(neuralNetType.equals(MULTI_LAYER_NETWORK)) {
                OptimizationConfiguration<DL4JConfiguration,MultiLayerNetwork,Object,Double> configuration = OptimizationConfiguration.
                        fromJson(
                                FileUtils.readFileToString(new File(optimizationConfigPath)),
                                DL4JConfiguration.class,
                                MultiLayerNetwork.class,
                                Object.class,
                                Double.class);

                IOptimizationRunner<DL4JConfiguration,MultiLayerNetwork,Double> runner
                        = new LocalOptimizationRunner<>(
                        configuration,
                        new MultiLayerNetworkTaskCreator<>(
                                new RegressionDataSetEvaluator(
                                        RegressionValue.valueOf(regressionType),
                                        commands)));
                runner.execute();
            }
        }

        else if(problemType.equals(CLASSIFICIATION)) {
            if(neuralNetType.equals(COMP_GRAPH)) {
                OptimizationConfiguration<GraphConfiguration,ComputationGraph,Object,Evaluation> configuration
                        = OptimizationConfiguration.fromJson(
                        FileUtils.readFileToString(new File(optimizationConfigPath)),
                        GraphConfiguration.class,
                        ComputationGraph.class,
                        Object.class,
                        Evaluation.class);

                IOptimizationRunner<GraphConfiguration,ComputationGraph,Evaluation> runner
                        = new LocalOptimizationRunner<>(
                        configuration,
                        new ComputationGraphTaskCreator<>(
                                new GraphClassificationDataSetEvaluator()
                        ));

                runner.execute();
            }
            else if(neuralNetType.equals(MULTI_LAYER_NETWORK)) {
                OptimizationConfiguration<DL4JConfiguration,MultiLayerNetwork,Object,Evaluation> configuration = OptimizationConfiguration
                        .fromJson(
                                FileUtils.readFileToString(new File(optimizationConfigPath)),
                                DL4JConfiguration.class,
                                MultiLayerNetwork.class,
                                Object.class,
                                Evaluation.class);

                IOptimizationRunner<DL4JConfiguration,MultiLayerNetwork,Evaluation> runner
                        = new LocalOptimizationRunner<>(configuration,
                        new MultiLayerNetworkTaskCreator<>(
                                new ClassificationEvaluator())
                );

                runner.execute();
            }
        }
    }
    public static void main(String...args) throws Exception {
        new ArbiterCliRunner().runMain(args);
    }

}
