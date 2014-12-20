package org.deeplearning4j.iterativereduce.impl;

import java.io.ByteArrayOutputStream;
import java.util.List;

import org.apache.commons.lang.time.StopWatch;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math.random.MersenneTwister;
import org.apache.commons.math.random.RandomGenerator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.iterativereduce.runtime.io.RecordParser;
import org.deeplearning4j.iterativereduce.runtime.io.TextRecordParser;
import org.deeplearning4j.iterativereduce.runtime.yarn.appworker.ApplicationWorker;


/**
 * Base IterativeReduce worker node
 *
 * @author josh
 *
 */
public class WorkerNode implements ComputableWorker<ParameterVectorUpdateable> {

    private static final Log LOG = LogFactory.getLog(WorkerNode.class);



    StopWatch watch = new StopWatch();
//	watch.start();




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
        return null;
    }



    @Override
    public ParameterVectorUpdateable compute(List<ParameterVectorUpdateable> arg0) {

        return compute();

    }

    @Override
    public ParameterVectorUpdateable getResults() {
        return null;
    }

    /**
     * TODO: re-wire this to read blocks of records into a Matrix
     *
     */
    @Override
    public void setRecordParser(RecordParser lineParser) {



    }

    /**
     * Setup the local DBN instance based on conf params
     *
     */
    @Override
    public void setup(Configuration c) {




        this.watch.start();


    }

    /**
     * Collect the update from the master node and apply it to the local
     * parameter vector
     *
     * TODO: check the state changes of the incoming message!
     *
     */
    @Override
    public void update(ParameterVectorUpdateable master_update_updateable) {




    }




    public static void main(String[] args) throws Exception {

        TextRecordParser parser = new TextRecordParser();
        WorkerNode wn = new WorkerNode();
        ApplicationWorker<ParameterVectorUpdateable> aw = new ApplicationWorker<>(parser, wn, ParameterVectorUpdateable.class);

        ToolRunner.run(aw, args);

    }



}