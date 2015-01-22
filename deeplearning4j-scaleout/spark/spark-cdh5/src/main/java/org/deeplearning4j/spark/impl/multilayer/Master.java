package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RDDMiniBatches;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.util.RDDUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Master class for spark
 *
 * @author Adam Gibson
 */
public class Master implements Serializable {

    private transient SparkContext sparkContext;
    private transient JavaSparkContext sc;
    private MultiLayerConfiguration conf;
    private RecordReader recordReader;



    public Master(SparkContext sparkContext,MultiLayerConfiguration conf,RecordReader recordReader) {
        this.sparkContext = sparkContext;
        this.conf = conf.clone();
        this.recordReader = recordReader;
        sc = new JavaSparkContext(this.sparkContext);
    }

    public Master(JavaSparkContext sc,MultiLayerConfiguration conf,RecordReader recordReader) {
        this.sc = sc;
        this.recordReader = recordReader;
        this.conf = conf.clone();
    }

    public MultiLayerNetwork fit(String path,int labelIndex) {
        JavaRDD<String> lines = sc.textFile(path);
        // gotta map this to a Matrix/INDArray
        JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(recordReader
                , labelIndex, conf.getConf(conf.getConfs().size() - 1).getnOut()));
        return fitDataSet(points);

    }


    public MultiLayerNetwork fit(JavaSparkContext sc,JavaRDD<LabeledPoint> rdd) {
        return fitDataSet(RDDUtil.fromLabeledPoint(sc,rdd,conf.getConf(conf.getConfs().size() - 1).getnOut()));
    }


    public MultiLayerNetwork fitDataSet(JavaRDD<DataSet> rdd) {
        //LabeledPoint
        //add dep on mllib
        //static runner

        int batchSize = conf.getConf(0).getBatchSize();
        JavaRDD<DataSet> miniBatches = new RDDMiniBatches(batchSize,rdd).miniBatchesJava();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray params = network.params();
        int paramsLength = network.numParams();
        if(params.length() != paramsLength)
            throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());
        INDArray newParams = miniBatches.map(new DL4jWorker(conf.toJson(),params)).reduce(new Function2<INDArray, INDArray, INDArray>() {
            @Override
            public INDArray call(INDArray v1, INDArray v2) throws Exception {
                return v1.add(v2);
            }
        }).divi(miniBatches.count());
        network.setParameters(newParams);
        return network;
    }
}
