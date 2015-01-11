package org.deeplearning4j.iterativereduce.impl.multilayer;

import java.io.IOException;
import java.util.List;



import org.apache.commons.lang3.time.StopWatch;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.impl.ParameterVectorUpdateable;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.iterativereduce.runtime.io.RecordParser;
import org.deeplearning4j.iterativereduce.runtime.io.SVMLightHDFSDataSetIterator;
import org.deeplearning4j.iterativereduce.runtime.io.TextRecordParser;
import org.deeplearning4j.iterativereduce.runtime.yarn.appworker.ApplicationWorker;

import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;




/**
 * Base IterativeReduce worker node
 *
 * @author josh
 *
 */
public class WorkerNode implements ComputableWorker<ParameterVectorUpdateable>,DeepLearningConfigurable {

    private MultiLayerNetwork multiLayerNetwork;
    private static Logger log = LoggerFactory.getLogger(WorkerNode.class);
    private TextRecordParser recordParser;
    private DataSetIterator hdfsDataSetIterator = null;
    private long totalRecordsProcessed = 0;
    StopWatch totalRunTimeWatch = new StopWatch();
    StopWatch batchWatch = new StopWatch();

    // confs (for now, may get rid of later)
    private int batchSize = 20;
    private int numberFeatures = 10;
    private int numberClasses = 2;


    /**
     *
     * - app.iteration.count
     * 		- indicates how many times we're going to call the workers
     *
     */
    @Override
    public ParameterVectorUpdateable compute() {
        log.info("Worker > Compute() -------------------------- ");
        int recordsProcessed = 0;
        DataSet hdfs_recordBatch = null; //this.hdfs_fetcher.next();

        if ( this.hdfsDataSetIterator.hasNext() ) {
            hdfs_recordBatch = this.hdfsDataSetIterator.next();
            if (hdfs_recordBatch.getFeatures().rows() > 0) {
                // calc stats on number records processed
                this.totalRecordsProcessed += hdfs_recordBatch.getFeatures().rows();
                batchWatch.reset();
                batchWatch.start();
                this.multiLayerNetwork.fit( hdfs_recordBatch );
                batchWatch.stop();
                log.info("Worker > Processed Total " + recordsProcessed + ", Batch Time " + batchWatch.toString() + " Total Time " + totalRunTimeWatch.toString());
            }
            else {
                // in case we get a blank line
                log.info("Worker > Idle pass, no records left to process");

            }

        }
        else {

            log.info("Worker > Idle pass, no records left to process");

        }


        return new ParameterVectorUpdateable(multiLayerNetwork.params());
    }



    @Override
    public ParameterVectorUpdateable compute(List<ParameterVectorUpdateable> arg0) {
        return compute();

    }

    @Override
    public ParameterVectorUpdateable getResults() {
        return new ParameterVectorUpdateable(multiLayerNetwork.params());
    }

    /**
     *
     *
     */
    @Override
    public void setRecordParser(RecordParser lineParser) {

        this.recordParser = (TextRecordParser) lineParser;

        // we're assuming SVMLight for current tests
        try {
            this.hdfsDataSetIterator = new SVMLightHDFSDataSetIterator( this.batchSize, 1, this.recordParser, this.numberFeatures, this.numberClasses );
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }



    }

    /**
     * Setup the local DBN instance based on conf params
     *
     *
     *
     */
    @Override
    public void setup(Configuration conf) {

        log.info("Worker-Conf: " + conf.get(MULTI_LAYER_CONF));
        
        MultiLayerConfiguration confMLN = new NeuralNetConfiguration.Builder()
        .list(4).hiddenLayerSizes(new int[]{3,2,2}).build();
        String json = confMLN.toJson();

        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( json ); // conf.get(MULTI_LAYER_CONF));
        multiLayerNetwork = new MultiLayerNetwork(conf2);


    }

    /**
     * Collect the update from the master node and apply it to the local
     * parameter vector
     *
     * TODO: check the state changes of the incoming message!
     *
     */
    @Override
    public void update(ParameterVectorUpdateable masterUpdateUpdateable) {
        multiLayerNetwork.setParameters(masterUpdateUpdateable.get());
    }




    /**
     * Dev note: this method seems complete 
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

        TextRecordParser parser = new TextRecordParser();
        WorkerNode wn = new WorkerNode();
        ApplicationWorker<ParameterVectorUpdateable> aw = new ApplicationWorker<>(parser, wn, ParameterVectorUpdateable.class);

        ToolRunner.run(aw, args);

    }


    @Override
    public void setup(org.deeplearning4j.nn.conf.Configuration conf) {

    }
}