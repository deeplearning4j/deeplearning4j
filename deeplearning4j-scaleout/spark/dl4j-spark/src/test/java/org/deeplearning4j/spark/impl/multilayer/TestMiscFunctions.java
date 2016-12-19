package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.api.java.JavaPairRDD;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 17/12/2016.
 */
public class TestMiscFunctions extends BaseSparkTest {

    @Test
    public void testFeedForwardWithKey(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(3).nOut(3).activation("softmax").build())
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSetIterator iter = new IrisDataSetIterator(150,150);
        DataSet ds = iter.next();


        List<INDArray> expected = new ArrayList<>();
        List<Tuple2<Integer,INDArray>> mapFeatures = new ArrayList<>();
        int count = 0;
        int arrayCount = 0;
        Random r = new Random(12345);
        while(count < 150){
            int exampleCount = r.nextInt(5)+1;  //1 to 5 inclusive examples
            if(count + exampleCount > 150) exampleCount = 150 - count;

            INDArray subset = ds.getFeatures().get(NDArrayIndex.interval(count,count+exampleCount), NDArrayIndex.all());

            expected.add(net.output(subset, false));
            mapFeatures.add(new Tuple2<>(arrayCount, subset));
            arrayCount++;
            count += exampleCount;
        }

        JavaPairRDD<Integer,INDArray> rdd = sc.parallelizePairs(mapFeatures);

        SparkDl4jMultiLayer multiLayer = new SparkDl4jMultiLayer(sc, net, null);
        Map<Integer,INDArray> map = multiLayer.feedForwardWithKey(rdd, 16).collectAsMap();

        for( int i=0; i<expected.size(); i++ ){
            INDArray exp = expected.get(i);
            INDArray act = map.get(i);

            assertEquals(exp, act);
        }
    }


    @Test
    public void testFeedForwardWithKeyGraph(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("in1", "in2")
                .addLayer("0", new DenseLayer.Builder().nIn(4).nOut(3).build(), "in1")
                .addLayer("1", new DenseLayer.Builder().nIn(4).nOut(3).build(), "in2")
                .addLayer("2", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(6).nOut(3).activation("softmax").build(), "0", "1")
                .setOutputs("2")
                .build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSetIterator iter = new IrisDataSetIterator(150,150);
        DataSet ds = iter.next();


        List<INDArray> expected = new ArrayList<>();
        List<Tuple2<Integer,INDArray[]>> mapFeatures = new ArrayList<>();
        int count = 0;
        int arrayCount = 0;
        Random r = new Random(12345);
        while(count < 150){
            int exampleCount = r.nextInt(5)+1;  //1 to 5 inclusive examples
            if(count + exampleCount > 150) exampleCount = 150 - count;

            INDArray subset = ds.getFeatures().get(NDArrayIndex.interval(count,count+exampleCount), NDArrayIndex.all());

            expected.add(net.outputSingle(false, subset, subset));
            mapFeatures.add(new Tuple2<>(arrayCount, new INDArray[]{subset,subset}));
            arrayCount++;
            count += exampleCount;
        }

        JavaPairRDD<Integer,INDArray[]> rdd = sc.parallelizePairs(mapFeatures);

        SparkComputationGraph graph = new SparkComputationGraph(sc, net, null);
        Map<Integer,INDArray[]> map = graph.feedForwardWithKey(rdd, 16).collectAsMap();

        for( int i=0; i<expected.size(); i++ ){
            INDArray exp = expected.get(i);
            INDArray act = map.get(i)[0];

            assertEquals(exp, act);
        }
    }

}
