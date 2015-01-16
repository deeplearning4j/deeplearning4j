package org.deeplearning4j.iterativereduce.impl.multilayer;

import java.util.List;



import org.apache.commons.lang3.time.StopWatch;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.RecordReader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.scaleout.api.ir.ParameterVectorUpdateable;
import org.deeplearning4j.iterativereduce.impl.reader.RecordReaderDataSetIterator;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;


import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
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
    private RecordReader recordParser;
    private DataSetIterator hdfsDataSetIterator = null;
    private long totalRecordsProcessed = 0;
    StopWatch totalRunTimeWatch = new StopWatch();
    StopWatch batchWatch = new StopWatch();

    // confs (for now, may get rid of later)
    private int batchSize = 20;
    private int numberClasses = 2;
    private int labelIndex = -1;
    public final static String LABEL_INDEX = "org.deeplearning4j.labelindex";


    /**
     *
     * - app.iteration.count
     * 		- indicates how many times we're going to call the workers
     *
     */
    @Override
    public ParameterVectorUpdateable compute() {
        log.info("Worker > Compute() -------------------------- ");

        DataSet hdfs_recordBatch = null; //this.hdfs_fetcher.next();

        if ( this.hdfsDataSetIterator.hasNext() ) {
            hdfs_recordBatch = this.hdfsDataSetIterator.next();
            if (hdfs_recordBatch.getFeatures().rows() > 0) {

                log.info("Rows: " + hdfs_recordBatch.numExamples() + ", inputs: " + hdfs_recordBatch.numInputs() + ", " + hdfs_recordBatch);


                // calc stats on number records processed
                this.totalRecordsProcessed += hdfs_recordBatch.getFeatures().rows();
                batchWatch.reset();
                batchWatch.start();
                this.multiLayerNetwork.fit( hdfs_recordBatch );
                batchWatch.stop();
                log.info("Worker > Processed Total " + this.totalRecordsProcessed + ", Batch Time " + batchWatch.toString() + " Total Time " + totalRunTimeWatch.toString());
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
     * @param lineParser
     */
    @Override
    public void setRecordReader(RecordReader lineParser) {

        this.recordParser = lineParser;
        this.hdfsDataSetIterator = new RecordReaderDataSetIterator(recordParser,null,batchSize,labelIndex,numberClasses);


    }

    /**
     * setup the local DBN instance based on conf params
     *
     *
     *
     */
    @Override
    public void setup(Configuration conf) {

        log.info("Worker-Conf: " + conf.get(MULTI_LAYER_CONF));

        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( conf.get(MULTI_LAYER_CONF));
        this.batchSize = conf2.getConf(0).getBatchSize();
        this.numberClasses = conf2.getConf(conf2.getConfs().size() - 1).getnOut();
        labelIndex = conf.getInt(LABEL_INDEX,-1);
        if(labelIndex < 0)
            throw new IllegalStateException("Illegal label index");
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





    @Override
    public void setup(org.deeplearning4j.nn.conf.Configuration conf) {

    }
}