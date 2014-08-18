package org.deeplearning4j.iterativereduce.actor.multilayer;


import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.core.conf.DeepLearningConfigurableDistributed;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperConfigurationRetriever;
import org.deeplearning4j.util.SerializationUtils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.DoubleOptionHandler;
import org.kohsuke.args4j.spi.IntOptionHandler;
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
 *        -fte number of fine tune epochs to train on (default: 100)
 *        -pte number of epochs for pretraining (default: 100)
 *        -r   seed value for the random number generator (default: 123)
 *        -ftl the starter fine tune learning rate (default: 0.1)
 *        -ptl  the starter fine tune learning rate (default: 0.1)
 *        -sp   number of inputs to split by default: 10
 *        -adg use adagrad or not: default value: true
 *        
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
	@Option(name="-o",usage="number of outputs for the network",handler=IntOptionHandler.class)
	protected int outputs;
	@Option(name="-fte",usage="number of fine tune epochs to train on (default: 100)",handler=IntOptionHandler.class)
	protected int finetuneEpochs = 100;
	@Option(name="-pte",usage="number of epochs for pretraining (default: 100)",handler=IntOptionHandler.class)
	protected int pretrainEpochs = 100;
	@Option(name="-r",usage="seed value for the random number generator (default: 123)",handler=IntOptionHandler.class)
	protected long rngSeed = 123;
	@Option(name="-k",usage="the k for rbms (default: 1)",handler=IntOptionHandler.class)
	protected int k = 1;
	@Option(name="-c",usage="corruption level (for denoising autoencoders) (default: 0.3)",handler=DoubleOptionHandler.class)
	protected double corruptionLevel = 0.3;
	@Option(name="-h",usage="the host to connect to as a master (default: 127.0.0.1)")
	protected String host = "localhost";
	@Option(name="-ftl",usage="the starter fine tune learning rate (default: 0.1)",handler=DoubleOptionHandler.class)
	protected double finetuneLearningRate = 0.1;
	@Option(name="-ptl",usage="the starter pretrain learning rate (default: 0.1)",handler=DoubleOptionHandler.class)
	protected double pretrainLearningRate = 0.1;
	@Option(name="-hl",usage="hidden layer sizes (comma separated list)")
	protected String hiddenLayerSizesOption;
	protected int[] hiddenLayerSizes = {300,300,300};
	@Option(name="-t",usage="type of worker")
	protected String type = "master";
	@Option(name="-ad",usage="address of master worker")
	protected String address;
	@Option(name="-sp",usage="number of inputs to split by default: 10")
	protected int split = 10;
	@Option(name="-data",usage="class to instantiate")
	protected String dataSet;
	@Option(name = "-datasetpath",usage="dataset path; eg if you save a dataset you can just point the network runner at a path rather than worrying about a class to instantiate")
	protected String dataPath;
	@Option(name="-e",usage="number of examples to train on: if unspecified will just train on everything found")
	protected int numExamples = -1;
	@Option(name="-adg",usage="use adagrad; default true")
	protected boolean useAdaGrad = true;
	@Option(name = "-stp",usage="state tracker port")
	protected int stateTrackerPort = -1;

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

		if(hiddenLayerSizesOption != null) {
			String[] split = hiddenLayerSizesOption.split(",");
			hiddenLayerSizes = new int[split.length];
			for(int i = 0; i < split.length; i++) {
				hiddenLayerSizes[i] = Integer.parseInt(split[i]);
			}

		}
	}





	@SuppressWarnings("unchecked")
	public void exec() throws Exception {

		//this is just a worker node: load everything from the master. All we should need is the ip of the master
		//to applyTransformToDestination everything up
		if(type != null && type.equals("worker"))  {
			log.info("Initializing conf from zookeeper at " + host);
			ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever(host, 2181, "master");
			Conf conf = retriever.retrieve();
			String address = conf.getMasterUrl();
            log.info("Creating hazel cast state tracker... " + conf.getStateTrackerConnectionString());
            HazelCastStateTracker stateTracker = new HazelCastStateTracker(conf.getStateTrackerConnectionString());

            log.info("Creating hazel cast via worker " + stateTracker.connectionString());
            runner = new ActorNetworkRunner(type,address);
            runner.setStateTracker(stateTracker);
			runner.setup(conf);
			retriever.close();

		}
		else {
			Conf conf = new Conf();


			getDataSet();
			conf.setMultiLayerClazz((Class<? extends BaseMultiLayerNetwork>) Class.forName(getClassForAlgorithm()));
			conf.setLayerSizes(hiddenLayerSizes);
			conf.setSplit(10);
			conf.setnIn(iter.inputColumns());
			conf.setnOut(iter.totalOutcomes());
			conf.setPretrainEpochs(pretrainEpochs);
			conf.setFinetuneEpochs(finetuneEpochs);
			conf.setSeed(rngSeed);
			conf.setPretrainLearningRate(pretrainLearningRate);
			conf.setUseAdaGrad(useAdaGrad);
			conf.setCorruptionLevel(corruptionLevel);
			conf.setSplit(split);
			conf.setK(k);
			conf.setFinetuneLearningRate(finetuneLearningRate);
			conf.setPretrainEpochs(pretrainEpochs);
			conf.setPretrainLearningRate(pretrainLearningRate);


			//run the master
			runner = new ActorNetworkRunner("master",iter);
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

	public String getData() {
		return dataSet;
	}

	protected void getDataSet() {
		if(type.equals("worker"))
			return;
		try {
			if(this.dataPath != null && this.dataSet != null)
				throw new IllegalStateException("Can't have both a data applyTransformToDestination and a dataset path defined");

			else if(dataPath != null) {
				DataSet data = SerializationUtils.readObject(new File(dataPath));
				this.iter = new ListDataSetIterator(data.asList(),split);
			}
			
			else if(this.numExamples >= 0 && this.split != 0) 
				this.iter = (DataSetIterator) Class.forName(dataSet).getConstructor(Integer.class,Integer.class).newInstance(split,numExamples);
			else
				this.iter = (DataSetIterator) Class.forName(dataSet).newInstance();


		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}


	protected String getClassForAlgorithm() {
		switch(algorithm) {
		case  "sda" :
			return "org.deeplearning4j.sda.StackedDenoisingAutoEncoder";

		case "dbn" : 
			return "org.deeplearning4j.dbn.DBN";
		case "cdbn":
			return "org.deeplearning4j.dbn.CDBN";
		}
		return null;
	}


	public boolean isDone() {
		return iter.hasNext();
	}


	/**
	 * @param args
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


	public int getFinetuneEpochs() {
		return finetuneEpochs;
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


	public double getFinetineLearningRate() {
		return finetuneLearningRate;
	}


	public double getPretrainLearningRate() {
		return pretrainLearningRate;
	}


	public String getHiddenLayerSizesOption() {
		return hiddenLayerSizesOption;
	}


	public int[] getHiddenLayerSizes() {
		return hiddenLayerSizes;
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
