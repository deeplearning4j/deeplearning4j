package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.single;


import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.DoubleOptionHandler;
import org.kohsuke.args4j.spi.IntOptionHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.DataSetIterator;
import com.ccc.deeplearning.datasets.iterator.impl.IrisDataSetIterator;
import com.ccc.deeplearning.datasets.iterator.impl.LFWDataSetIterator;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.ExtraParamsBuilder;
import com.ccc.deeplearning.scaleout.core.conf.DeepLearningConfigurableDistributed;
import com.ccc.deeplearning.scaleout.zookeeper.ZooKeeperConfigurationRegister;
import com.ccc.deeplearning.scaleout.zookeeper.ZookeeperConfigurationRetriever;
/**
 * Main command line app for handling workers or starting up a master for training a neural network:
 * 
 * Options:
 *       Required:
 *       -a algorithm to use: sda (stacked denoising autoencoders),dbn (deep belief networks),cdbn (continuous deep belief networks)
 *       -i number of inputs (columns in the input matrix)
 *       -o number of outputs for the network
 *       -data dataset to train on: options: mnist,text (text files with <label>text</label>, image (images where the parent directory is the label)
 *       
 *       Optional:
 *        -fte number of fine tune epochs to train on (default: 100)
 *        -pte number of epochs for pretraining (default: 100)
 *        -r   seed value for the random number generator (default: 123)
 *        -ftl the starter fine tune learning rate (default: 0.1)
 *        -ptl  the starter fine tune learning rate (default: 0.1)
 *        -sp   number of inputs to split by default: 10
 *        -e   number of examples to train on: if unspecified will just train on everything found
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
public class ActorNetworkRunnerApp implements DeepLearningConfigurableDistributed {
	protected static Logger log = LoggerFactory.getLogger(ActorNetworkRunnerApp.class);

	@Option(name = "-a",usage="algorithm to use: sda (stacked denoising autoencoders),dbn (deep belief networks),cdbn (continuous deep belief networks)")
	protected String algorithm;
	@Option(name = "-i",usage="number of inputs (columns in the input matrix)",handler=IntOptionHandler.class)
	protected int inputs;
	@Option(name="-o",usage="number hidden units for the network",handler=IntOptionHandler.class)
	protected int outputs;
	@Option(name="-pte",usage="number of epochs for pretraining (default: 100)",handler=IntOptionHandler.class)
	protected int pretrainEpochs = 1;
	@Option(name="-r",usage="seed value for the random number generator (default: 123)",handler=IntOptionHandler.class)
	protected long rngSeed = 123;
	@Option(name="-k",usage="the k for rbms (default: 1)",handler=IntOptionHandler.class)
	protected int k = 1;
	@Option(name="-c",usage="corruption level (for denoising autoencoders) (default: 0.3)",handler=DoubleOptionHandler.class)
	protected double corruptionLevel = 0.3;
	@Option(name="-h",usage="the host to connect to as a master (default: 127.0.0.1)")
	protected String host = "localhost";
	@Option(name="-ptl",usage="the starter pretrain learning rate (default: 0.1)",handler=DoubleOptionHandler.class)
	protected double pretrainLearningRate = 0.1;
	@Option(name="-t",usage="type of worker")
	protected String type = "master";
	@Option(name="-ad",usage="address of master worker")
	protected String address;
	@Option(name="-sp",usage="number of inputs to split by default: 10")
	protected int split = 10;
	@Option(name="-data",usage="dataset to train on: options: mnist,text (text files with <label>text</label>, image (images where the parent directory is the label)")
	protected String dataSet;
	@Option(name="-e",usage="number of examples to train on: if unspecified will just train on everything found")
	protected int numExamples = -1;
	@Option(name="-l2",usage="l2 regularization constant")
	protected double l2 = 0.1;
	@Option(name="-m",usage="momentum")
	protected double momentum = 0.1;

	protected ActorNetworkRunner runner;
	protected DataSetIterator iter;


	public ActorNetworkRunnerApp(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			parser.printUsage(System.err);
			log.error("Unable to parse args",e);
		}


	}





	public void exec() throws Exception {

		//this is just a worker node: load everything from the master. All we should need is the ip of the master
		//to set everything up
		if(type != null && type.equals("worker"))  {
			log.info("Initializing conf from zookeeper at " + host);
			ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever(host, 2181, "master");
			Conf conf = retriever.retreive();
			String address = conf.getMasterUrl();
			runner = new ActorNetworkRunner(type,address);
			runner.setup(conf);
			retriever.close();

		}
		else {
			Conf conf = new Conf();


			getDataSet();
			conf.setMultiLayerClazz((Class<? extends BaseMultiLayerNetwork>) Class.forName(getClassForAlgorithm()));
			conf.setSplit(10);
			if(inputs < 1)
				conf.setnIn(iter.inputColumns());
			else
				conf.setnIn(inputs);
			if(outputs < 1)
				conf.setnOut(iter.totalOutcomes());
			else
				conf.setnOut(outputs);
			conf.setPretrainEpochs(pretrainEpochs);
			conf.setSeed(rngSeed);
			conf.setPretrainLearningRate(pretrainLearningRate);
			conf.setL2(l2);
			conf.setMomentum(momentum);
			conf.setCorruptionLevel(corruptionLevel);
			conf.setSplit(split);
			conf.setK(k);
			
			//run the master
			runner = new ActorNetworkRunner("master",iter);
			runner.setup(conf);
			//store it in zookeeper for service discovery
			conf.setMasterUrl(runner.getMasterAddress().toString());

			//register the configuration to zookeeper
			ZooKeeperConfigurationRegister reg = new ZooKeeperConfigurationRegister(conf,"master",host,2181);
			reg.register();
			reg.close();



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

		DataSet batch = null;
		int numBatches = 0;
		//trainer will handle batching for the rest 
		//based on responding to actors
		if(iter.hasNext()) {
			log.info("Training next batch " + (numBatches + 1));
			batch = iter.next();
			runner.train(batch);
			numBatches++;
		}

	}


	public void shutdown() {

	}

	public String getData() {
		return dataSet;
	}

	protected void getDataSet() {
		if(type.equals("worker"))
			return;
		try {
			if(dataSet.equals("mnist")) {
				iter = new MnistDataSetIterator(split,numExamples);
			}

			else if(dataSet.equals("iris")) {
				iter = new IrisDataSetIterator(split,numExamples);
			}
			else if(dataSet.equals("lfw")) {
				iter = new LFWDataSetIterator(split,numExamples);
			}

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}


	protected String getClassForAlgorithm() {
		switch(algorithm) {
		case  "da" :
			return "com.ccc.deeplearning.sda.DenoisingAutoEncoder";

		case "rbm" : 
			return "com.ccc.deeplearning.rbm.RBM";
		case "crbm":
			return "com.ccc.deeplearning.rbm.CRBM";
		}
		return null;
	}


	public boolean isDone() {
		return iter.hasNext();
	}


	/**
	 * @param args
	 * @throws ParseException 
	 */
	public static void main(String[] args) throws Exception {
		ActorNetworkRunnerApp app = new ActorNetworkRunnerApp(args);
		app.exec();
		if(app.type.equals("master"))
			app.train();
	}

	@Override
	public void setup(Conf conf) {

	}


	public String getAlgorithm() {
		return algorithm;
	}


	public int getInputs() {
		return inputs;
	}


	public int getOutputs() {
		return outputs;
	}




	public int getPretrainEpochs() {
		return pretrainEpochs;
	}


	public long getRngSeed() {
		return rngSeed;
	}


	public int getK() {
		return k;
	}


	public double getCorruptionLevel() {
		return corruptionLevel;
	}


	public String getHost() {
		return host;
	}



	public double getPretrainLearningRate() {
		return pretrainLearningRate;
	}



	public String getType() {
		return type;
	}


	public String getAddress() {
		return address;
	}


	public int getSplit() {
		return split;
	}


	public int getNumExamples() {
		return numExamples;
	}


}
