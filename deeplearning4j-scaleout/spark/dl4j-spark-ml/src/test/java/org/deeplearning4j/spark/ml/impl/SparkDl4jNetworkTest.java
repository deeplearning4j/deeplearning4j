package org.deeplearning4j.spark.ml.impl;


import com.google.common.collect.Lists;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.ml.utils.ParamSerializer;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SparkDl4jNetworkTest  {

    private SparkConf sparkConf = new SparkConf().setAppName("testing").setMaster("local[4]");
    private JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
    private SQLContext sqlContext = new SQLContext(sparkContext);

    @Test
    public void testNetwork() {
        List<Row> dubs = Lists.newArrayList(
                RowFactory.create(1000.0, 1000.0, 1.0),
                RowFactory.create(90.0, 90.0, 0.0)
        );
        DataFrame df = sqlContext.createDataFrame(dubs, createStruct());
        Pipeline p = new Pipeline().setStages(new PipelineStage[]{getAssembler(new String[]{"x", "y"}, "features")});
        DataFrame part2 = p.fit(df).transform(df).select("features", "label");

        ParamSerializer ps = new ParamHelper();
        MultiLayerConfiguration mc = getNNConfiguration();
        Collection<IterationListener> il = new ArrayList<>();
        il.add(new ScoreIterationListener(1));

        SparkDl4jNetwork sparkDl4jNetwork = new SparkDl4jNetwork(mc, 2, ps, il)
                .setFeaturesCol("features")
                .setLabelCol("label");

        SparkDl4jModel sm = sparkDl4jNetwork.fit(part2);
        MultiLayerNetwork mln = sm.getMultiLayerNetwork();
        Assert.assertNotNull(mln);
        System.out.println(sm.output(Vectors.dense(0.0, 0.0)));
    }

    private StructType createStruct() {
        return  new StructType(new StructField[]{
                new StructField("x", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("y", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, true, Metadata.empty())
        });
    }

    private MultiLayerConfiguration getNNConfiguration(){
        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(10)
                .weightInit(WeightInit.UNIFORM)
                .learningRate(0.1)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(100).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1,  new DenseLayer.Builder().nIn(100).nOut(120).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("softmax").nIn(120).nOut(2).build())
                .pretrain(false).backprop(true)
                .build();
    }

    public static VectorAssembler getAssembler(String[] input, String output){
        return new VectorAssembler()
                .setInputCols(input)
                .setOutputCol(output);
    }

    static public class ParamHelper implements ParamSerializer {

        public ParameterAveragingTrainingMaster apply() {
            return new ParameterAveragingTrainingMaster.Builder(2)
                    .averagingFrequency(2)
                    .workerPrefetchNumBatches(2)
                    .batchSizePerWorker(2)
                    .build();
        }
    }
}
