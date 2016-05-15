package org.deeplearning4j;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.cli.subcommands.BaseSubCommand;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.impl.computationgraph.SparkComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;

/**
 * Spark train command on spark
 *
 * @author Adam Gibson
 */
public class SparkTrain extends BaseSubCommand {
    @Option(name = "--model", usage = "model file (json,yaml,..) to resume training",aliases = "-mo", required = true)
    private String modelInput;
    @Option(name = "--conf", usage = "computation graph configuration",aliases = "-c", required = true)
    private String confInput;
    @Option(name = "--masterUri", usage = "spark master uri",aliases = "-ma", required = true)
    private String masterUri;
    @Option(name = "--input", usage = "input data",aliases = "-i", required = true)
    private String masterInputUri;
    @Option(name = "--type", usage = "input data type",aliases = "-t", required = true)
    private String inputType;
    @Option(name = "--examplesPerFit", usage = "examples per fit",aliases = "-b", required = true)
    private int examplesPerFit;
    @Option(name = "--totalExamples", usage = "total number of examples",aliases = "-n", required = true)
    private int totalExamples;
    @Option(name = "--numPartitions", usage = "number of partitions",aliases = "-p", required = true)
    private int numPartitions;
    @Option(name = "--output", usage = "output path",aliases = "-o", required = true)
    private String outputPath;
    private SparkContext sc;

    /**
     * @param args arguments for command
     */
    public SparkTrain(String[] args) {
        super(args);
    }

    private SparkContext getContext() {
        if(sc != null)
            return sc;
        return null;
    }

    private JavaRDD<DataSet> getDataSet() {
        SparkContext sc = getContext();
        if (inputType.equals("binary")) {

        }
        else if(inputType.equals("text")) {

        }
        else
            throw new IllegalArgumentException("Input type must be either binary or text.");
        return null;
    }

    private ComputationGraph getComputationGraph() throws IOException {
        if(confInput != null &&  modelInput != null)
            throw new IllegalArgumentException("Conf and model input both can't be defined");
        ComputationGraph graph = null;

        if(confInput != null) {
            ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(confInput);
            graph = new ComputationGraph(conf);
            graph.init();

        }
        else if(modelInput != null) {
            graph = ModelSerializer.restoreComputationGraph(modelInput);

        }

        return graph;
    }

    private void saveGraph(ComputationGraph graph) {

    }




    /**
     * Execute a command
     */
    @Override
    public void execute() {
        ComputationGraph graph;

        try {
            graph = getComputationGraph();
            SparkComputationGraph multiLayer = new SparkComputationGraph(getContext(),graph);
            JavaRDD<DataSet> dataSet = getDataSet();
            //int examplesPerFit, int totalExamples, int numPartitions
            ComputationGraph newGraph = multiLayer.fitDataSet(dataSet,examplesPerFit,totalExamples,numPartitions);
            saveGraph(newGraph);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
