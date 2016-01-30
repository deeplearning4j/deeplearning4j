package org.deeplearning4j.spark.impl.graph;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.serializer.SerializationDebugger;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.computationgraph.SparkComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

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


}
