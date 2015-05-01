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

package org.deeplearning4j.scaleout.actor.runner;



import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.job.JobIteratorFactory;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperConfigurationRetriever;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Main command line app for handling workers or starting up a master for training a neural network:
 * TODO: Add classification
 * 
 * Options:
 *       Required:
 *       -a algorithm to use: sda (stacked denoising autoencoders),dbn (deep belief networks),cdbn (continuous deep belief networks)
 *       -i number of inputs (columns in the input matrix)
 *       -o number of outputs for the network
 *       
 *       DataSets:
 *         Note only one of these may be specified
 *         -data fully qualified class name of the dataset iterator to use
 *         -datasetpath path to a serialized dataset
 *          
 *       
 *       Optional:
 *        -fte number of fine tune epochs to iterate on (default: 100)
 *        -pte number of epochs for pretraining (default: 100)
 *        -r   seed value for the random number generator (default: 123)
 *        -ftl the starter fine tune learning rate (default: 0.1)
 *        -ptl  the starter fine tune learning rate (default: 0.1)
 *        -sp   number of inputs to split by default: 10
 *        -adg use adagrad or not: default value: true
 *        
 *        -e   number of examples to iterate on: if unspecified will just iterate on everything found
 *        DBN/CDBN:
 *        -k the k for rbms (default: 1)
 *        
 *        SDA:
 *        -c corruption level (for denoising autoencoders) (default: 0.3)
 *        
 *        Cluster:
 *        
 *            -h the host to connect to as a master (default: 127.0.0.1)
 *            -t type of worker
 *            -ad address of master worker
 *        
 *      
 * @author Adam Gibson
 *
 */
public class DeepLearning4jDistributedApp implements DeepLearningConfigurable {
	protected static final Logger log = LoggerFactory.getLogger(DeepLearning4jDistributedApp.class);

	@Option(name="-h",usage="the host to connect to as a master (default: 127.0.0.1)")
	protected String host = "localhost";
	@Option(name="-t",usage="type of worker")
	protected String type = "master";
	@Option(name="-ad",usage="address of master worker")
	protected String address;
	@Option(name = "-stp",usage="state tracker port")
	protected int stateTrackerPort = -1;
    @Option(name = "-jsonpath",usage = "specify a path to a json file")
    protected String jsonPath;
    @Option(name = "-json",usage = "json for configuration")
    protected String json;
	protected DeepLearning4jDistributed runner;
	protected JobIterator iter;
    @Option(name = "-jobclass",usage = "job class")
    protected String jobFactoryClazz;

	public DeepLearning4jDistributedApp(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			parser.printUsage(System.err);
			log.error("Unable to parse args",e);
		}

	}

	@SuppressWarnings("unchecked")
	public void exec() throws Exception {

		//this is just a worker node: load everything from the master. All we should need is the ip of the master
		//to applyTransformToDestination everything up
		if(type != null && type.equals("worker"))  {
			log.info("Initializing conf from zookeeper at " + host);
			ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever(host, 2181, "master");
			Configuration conf = retriever.retrieve();
			String address = conf.get(MASTER_URL);
            log.info("Creating hazel cast state tracker... " + conf.get(STATE_TRACKER_CONNECTION_STRING));
            HazelCastStateTracker stateTracker = new HazelCastStateTracker(conf.get(STATE_TRACKER_CONNECTION_STRING));

            log.info("Creating hazel cast via worker " + stateTracker.connectionString());
            runner = new DeepLearning4jDistributed(type,address);
            runner.setMasterHost(host);
            runner.setStateTracker(stateTracker);
			runner.setup(conf);
			retriever.close();

		}
		else {
            Configuration conf = new Configuration();
            JobIteratorFactory factory = (JobIteratorFactory) Class.forName(jobFactoryClazz).newInstance();
            iter = factory.create();

			//run the master
			runner = new DeepLearning4jDistributed("master",iter);
			runner.setStateTrackerPort(stateTrackerPort);
			runner.setup(conf);
		}
	}

	/**
	 * Initializes the training.
	 * Note that this is only used as a trigger for the initial call.
	 * The ActorSystem already has a work pull pattern implemented 
	 * with a batch actor, and a reference to the iterator.
	 * 
	 */
	public void train() {
       runner.train();
	}

	public void shutdown() {

	}

	public boolean isDone() {
		return iter.hasNext();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		DeepLearning4jDistributedApp app = new DeepLearning4jDistributedApp(args);
		app.exec();
		if(app.type.equals("master"))
			app.train();
	}

	@Override
	public void setup(Configuration conf) {

	}





}
