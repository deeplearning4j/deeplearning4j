package org.deeplearning4j.spark.impl.graph;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.computationgraph.SparkComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class TestSparkComputationGraph extends BaseSparkTest {

    @Test
    public void testBasic() throws Exception {

        JavaSparkContext sc = this.sc;

        RecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        MultiDataSetIterator iter = new RecordReaderMultiDataSetIterator.Builder(1)
                .addReader("iris",rr)
                .addInput("iris",0,3)
                .addOutputOneHot("iris",4,3)
                .build();

        List<MultiDataSet> list = new ArrayList<>(150);
        while(iter.hasNext()) list.add(iter.next());

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.1)
                .graphBuilder()
                .addInputs("in")
                .addLayer("dense",new DenseLayer.Builder().nIn(4).nOut(2).build(),"in")
                .addLayer("out",new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3).build(),"dense")
                .setOutputs("out")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        SparkComputationGraph scg = new SparkComputationGraph(sc,cg);

        JavaRDD<MultiDataSet> rdd = sc.parallelize(list);
        scg.fitMultiDataSet(rdd, 10, 150, 10);

        //Try: fitting using DataSet
        DataSetIterator iris = new IrisDataSetIterator(1,150);
        List<DataSet> list2 = new ArrayList<>();
        while(iris.hasNext()) list2.add(iris.next());
        JavaRDD<DataSet> rddDS = sc.parallelize(list2);

        scg.fitDataSet(rddDS,10,150,15);
    }


    @Test
    public void testDistributedScoring(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .regularization(true).l1(0.1).l2(0.1)
                .seed(123)
                .updater(Updater.NESTEROVS)
                .learningRate(0.1)
                .momentum(0.9)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(nIn).nOut(3)
                        .activation("tanh").build(),"in")
                .addLayer("1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(nOut)
                        .activation("softmax")
                        .build(),"0")
                .setOutputs("1")
                .backprop(true)
                .pretrain(false)
                .build();

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc,conf);
        ComputationGraph netCopy = sparkNet.getNetwork().clone();

        int nRows = 100;

        INDArray features = Nd4j.rand(nRows, nIn);
        INDArray labels = Nd4j.zeros(nRows, nOut);
        Random r = new Random(12345);
        for( int i=0; i<nRows; i++ ){
            labels.putScalar(new int[]{i,r.nextInt(nOut)},1.0);
        }

        INDArray localScoresWithReg = netCopy.scoreExamples(new DataSet(features,labels),true);
        INDArray localScoresNoReg = netCopy.scoreExamples(new DataSet(features,labels),false);

        List<Tuple2<String,DataSet>> dataWithKeys = new ArrayList<>();
        for( int i=0; i<nRows; i++ ){
            DataSet ds = new DataSet(features.getRow(i).dup(),labels.getRow(i).dup());
            dataWithKeys.add(new Tuple2<>(String.valueOf(i),ds));
        }
        JavaPairRDD<String,DataSet> dataWithKeysRdd = sc.parallelizePairs(dataWithKeys);

        JavaPairRDD<String,Double> sparkScoresWithReg = sparkNet.scoreExamplesDataSet(dataWithKeysRdd, true, 4);
        JavaPairRDD<String,Double> sparkScoresNoReg = sparkNet.scoreExamplesDataSet(dataWithKeysRdd,false,4);

        Map<String,Double> sparkScoresWithRegMap = sparkScoresWithReg.collectAsMap();
        Map<String,Double> sparkScoresNoRegMap = sparkScoresNoReg.collectAsMap();

        for( int i=0; i<nRows; i++ ){
            double scoreRegExp = localScoresWithReg.getDouble(i);
            double scoreRegAct = sparkScoresWithRegMap.get(String.valueOf(i));
            assertEquals(scoreRegExp,scoreRegAct,1e-5);

            double scoreNoRegExp = localScoresNoReg.getDouble(i);
            double scoreNoRegAct = sparkScoresNoRegMap.get(String.valueOf(i));
            assertEquals(scoreNoRegExp, scoreNoRegAct, 1e-5);

//            System.out.println(scoreRegExp + "\t" + scoreRegAct + "\t" + scoreNoRegExp + "\t" + scoreNoRegAct);
        }

        List<DataSet> dataNoKeys = new ArrayList<>();
        for( int i=0; i<nRows; i++ ){
            dataNoKeys.add(new DataSet(features.getRow(i).dup(),labels.getRow(i).dup()));
        }
        JavaRDD<DataSet> dataNoKeysRdd = sc.parallelize(dataNoKeys);

        List<Double> scoresWithReg = sparkNet.scoreExamplesDataSet(dataNoKeysRdd,true,4).collect();
        List<Double> scoresNoReg = sparkNet.scoreExamplesDataSet(dataNoKeysRdd,false,4).collect();
        Collections.sort(scoresWithReg);
        Collections.sort(scoresNoReg);
        double[] localScoresWithRegDouble = localScoresWithReg.data().asDouble();
        double[] localScoresNoRegDouble = localScoresNoReg.data().asDouble();
        Arrays.sort(localScoresWithRegDouble);
        Arrays.sort(localScoresNoRegDouble);

        for( int i=0; i<localScoresWithRegDouble.length; i++ ){
            assertEquals(localScoresWithRegDouble[i],scoresWithReg.get(i),1e-5);
            assertEquals(localScoresNoRegDouble[i],scoresNoReg.get(i),1e-5);

//            System.out.println(localScoresWithRegDouble[i] + "\t" + scoresWithReg.get(i) + "\t" + localScoresNoRegDouble[i] + "\t" + scoresNoReg.get(i));
        }
    }
}
