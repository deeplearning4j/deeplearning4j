package org.deeplearning4j.spark.impl.layer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RDDMiniBatches;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
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
    private NeuralNetConfiguration conf;
    private RecordReader recordReader;



    public Master(SparkContext sparkContext,NeuralNetConfiguration conf,RecordReader recordReader) {
        this.sparkContext = sparkContext;
        this.conf = conf.clone();
        this.recordReader = recordReader;
        sc = new JavaSparkContext(this.sparkContext);
    }

    public Master(JavaSparkContext sc,NeuralNetConfiguration conf,RecordReader recordReader) {
        this.sc = sc;
        this.recordReader = recordReader;
        this.conf = conf.clone();
    }

    public Layer fit(String path,int labelIndex,int numLabels) {
        JavaRDD<String> lines = sc.textFile(path);
        // gotta map this to a Matrix/INDArray
        JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(recordReader
                , labelIndex, numLabels));
        return fit(points);

    }

    public Layer fit(JavaRDD<DataSet> rdd) {
        int batchSize = conf.getBatchSize();
        JavaRDD<DataSet> miniBatches = new RDDMiniBatches(batchSize,rdd).miniBatchesJava();
        Layer layer = conf.getLayerFactory().create(conf);
        INDArray params = layer.params();
        int paramsLength = layer.numParams();
        if(params.length() != paramsLength)
            throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());
        INDArray newParams = miniBatches.map(new DL4jWorker(conf.toJson(),params)).reduce(new Function2<INDArray, INDArray, INDArray>() {
            @Override
            public INDArray call(INDArray v1, INDArray v2) throws Exception {
                return v1.add(v2);
            }
        }).divi(miniBatches.count());
        layer.setParameters(newParams);
        return layer;
    }
}
