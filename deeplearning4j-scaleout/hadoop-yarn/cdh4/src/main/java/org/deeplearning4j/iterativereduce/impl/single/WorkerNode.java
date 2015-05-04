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

package org.deeplearning4j.iterativereduce.impl.single;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.RecordReader;
import org.deeplearning4j.scaleout.api.ir.ParameterVectorUpdateable;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;


/**
 * Base IterativeReduce worker node
 *
 * @author josh
 *
 */
public class WorkerNode implements ComputableWorker<ParameterVectorUpdateable>,DeepLearningConfigurable {

    private static final Logger LOG = LoggerFactory.getLogger(WorkerNode.class);
    private Layer neuralNetwork;
    private RecordReader recordParser;


    /**
     * Run a training pass of a single batch of input records on the DBN
     *
     * TODO
     * - dileneate between pre-train and finetune pass through data
     * 		- how?
     *
     * - app.iteration.count
     * 		- indicates how many times we're going to call the workers
     *
     * - tv.floe.metronome.dbn.conf.batchSize=10
     * 		- indicates that we're going to only process 10 records in a call to a worker
     *
     * - we could either
     *
     * 		1. make a complete pass through the batches in a split between iterations
     *
     * 			- tends to skew away from good solutions
     *
     * 		2. parameter average between batches
     *
     *			-	better quality, but more network overhead
     *
     * - if we paramete avg between batches, then our passes over the dataset become
     *
     * 		- total number of examples / batch size
     *
     * - might be pragmatic to let a command line tool calculate iterations
     *
     * 		- given we need to know how many fine tune passes to make as well
     *
     *
     *
     *
     *
     *
     *
     *
     */
    @Override
    public ParameterVectorUpdateable compute() {
        try {
            while(recordParser.nextKeyValue()) {
                DataSet params = (DataSet) recordParser.getCurrentValue();
                neuralNetwork.fit(params.getFeatureMatrix());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new ParameterVectorUpdateable(neuralNetwork.params());
    }

    @Override
    public void setRecordReader(org.apache.hadoop.mapreduce.RecordReader r) {
        this.recordParser = r;
    }



    @Override
    public ParameterVectorUpdateable compute(List<ParameterVectorUpdateable> arg0) {

        return compute();

    }

    @Override
    public ParameterVectorUpdateable getResults() {
        return new ParameterVectorUpdateable(neuralNetwork.params());
    }



    /**
     * setup the local DBN instance based on conf params
     *
     */
    @Override
    public void setup(Configuration conf) {
        NeuralNetConfiguration conf2 = NeuralNetConfiguration.fromJson(conf.get(NEURAL_NET_CONF));

        neuralNetwork = LayerFactories.getFactory(conf2.getLayer()).create(conf2);


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
        neuralNetwork.setParams(masterUpdateUpdateable.get());
    }





    @Override
    public void setup(org.canova.api.conf.Configuration conf) {

    }
}