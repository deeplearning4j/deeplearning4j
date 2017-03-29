package org.deeplearning4j.spark.ml.impl;


import com.google.common.collect.Lists;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.ml.utils.ParamSerializer;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class AutoEncoderNetworkTest {

    private SparkConf sparkConf = new SparkConf().setAppName("testing").setMaster("local[4]");
    private JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
    private SQLContext sqlContext = new SQLContext(sparkContext);

    @Test
    public void testNetwork() {
        List<Row> dubs = Lists.newArrayList(
                RowFactory.create(Vectors.dense(100., 125., 644., 3432., 1233., 5435., 7675., 32423., 6456457.,2., 4, 6., 1.0)),
                RowFactory.create(Vectors.dense(100., 165., 6234., 3132., 1533., 54365., 762345., 1423., 56457.,2.3, 41, 64., 16.0))
        );
        DataFrame df = sqlContext.createDataFrame(dubs, createStruct());
        Pipeline p = new Pipeline().setStages(new PipelineStage[]{getAssembler(new String[]{"x"}, "features")});
        DataFrame part2 = p.fit(df).transform(df).select("features");

        AutoEncoder sparkDl4jNetwork = new AutoEncoder()
                .setInputCol("features")
                .setOutputCol("auto_encoded")
                .setCompressedLayer(2)
                .setTrainingMaster(new ParamHelper())
                .setMultiLayerConfiguration(getNNConfiguration());

        AutoEncoderModel sm = sparkDl4jNetwork.fit(part2);
        DataFrame d2 = sm.transform(part2);
        MultiLayerNetwork mln = sm.getNetwork();
        Assert.assertNotNull(mln);
    }

    private StructType createStruct() {
        return  new StructType(new StructField[]{
                new StructField("x", new VectorUDT(), true, Metadata.empty())
        });
    }

    private MultiLayerConfiguration getNNConfiguration(){
        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(5)
                .learningRate(.9)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(13).nOut(9).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(1, new RBM.Builder().nIn(9).nOut(5).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(2, new RBM.Builder().nIn(5).nOut(2).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(3, new RBM.Builder().nIn(2).nOut(5).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(4, new RBM.Builder().nIn(5).nOut(9).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //decoding starts
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("sigmoid").nIn(9).nOut(13).build())
                .pretrain(true).backprop(true)
                .build();
    }

    public static VectorAssembler getAssembler(String[] input, String output){
        return new VectorAssembler()
                .setInputCols(input)
                .setOutputCol(output);
    }

    static public class ParamHelper implements ParamSerializer {

        public ParameterAveragingTrainingMaster apply() {
            return new ParameterAveragingTrainingMaster.Builder(3)
                    .averagingFrequency(2)
                    .workerPrefetchNumBatches(2)
                    .batchSizePerWorker(2)
                    .build();
        }
    }
}
