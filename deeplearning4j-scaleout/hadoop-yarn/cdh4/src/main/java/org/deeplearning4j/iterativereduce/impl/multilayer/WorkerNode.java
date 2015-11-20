/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.iterativereduce.impl.multilayer;

import java.util.List;



import org.apache.commons.lang3.time.StopWatch;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.RecordReader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.impl.reader.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.scaleout.api.ir.ParameterVectorUpdateable;
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
    private static final Logger log = LoggerFactory.getLogger(WorkerNode.class);
    private RecordReader recordParser;
    private DataSetIterator hdfsDataSetIterator = null;
    private long totalRecordsProcessed = 0;
    private StopWatch totalRunTimeWatch = new StopWatch();
    private StopWatch batchWatch = new StopWatch();

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

        DataSet hdfsBatch = null;

        if ( this.hdfsDataSetIterator.hasNext() ) {
            hdfsBatch = this.hdfsDataSetIterator.next();
            if (hdfsBatch.getFeatures().rows() > 0) {

                log.info("Rows: " + hdfsBatch.numExamples() + ", inputs: " + hdfsBatch.numInputs() + ", " + hdfsBatch);


                // calc stats on number records processed
                this.totalRecordsProcessed += hdfsBatch.getFeatures().rows();
                batchWatch.reset();
                batchWatch.start();
                this.multiLayerNetwork.fit( hdfsBatch );
                batchWatch.stop();
                log.info("Worker > Processed Total " + this.totalRecordsProcessed + ", Batch Time " + batchWatch.toString() + " Total Time " + totalRunTimeWatch.toString());
            }
            else {
                // in case we get a blank line
                log.info("Worker > Idle pass, no records left to process");

            }

        }

        else
            log.info("Worker > Idle pass, no records left to process");




        return new ParameterVectorUpdateable(multiLayerNetwork.params());
    }

    @Override
    public void setRecordReader(RecordReader r) {
        this.recordParser = r;
        // int batchSize,int labelIndex,int numPossibleLabels
        this.hdfsDataSetIterator = new RecordReaderDataSetIterator(r,null,batchSize,labelIndex,numberClasses);
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
     * setup the local DBN instance based on conf params
     *
     *
     *
     */
    @Override
    public void setup(Configuration conf) {

        log.info("Worker-Conf: " + conf.get(MULTI_LAYER_CONF));

        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( conf.get(MULTI_LAYER_CONF));
        FeedForwardLayer outputLayer = (FeedForwardLayer) conf2.getConf(conf2.getConfs().size() - 1).getLayer();
        this.numberClasses = outputLayer.getNOut();
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
    public void setup(org.canova.api.conf.Configuration conf) {

    }
}