package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.DoubleOptionHandler;
import org.kohsuke.args4j.spi.IntOptionHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.DataSetIterator;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl.IrisDataSetIterator;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl.LFWDataSetIterator;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZooKeeperConfigurationRegister;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZookeeperConfigurationRetriever;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurableDistributed;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.ExtraParamsBuilder;
/**
 * Main command line app for handling workers or starting up a master for training a neural network:
 * TODO: Add classification
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
	private static Logger log = LoggerFactory.getLogger(ActorNetworkRunnerApp.class);

	@Option(name = "-a",usage="algorithm to use: sda (stacked denoising autoencoders),dbn (deep belief networks),cdbn (continuous deep belief networks)")
	private String algorithm;
	@Option(name = "-i",usage="number of inputs (columns in the input matrix)",handler=IntOptionHandler.class)
	private int inputs;
	@Option(name="-o",usage="number of outputs for the network",handler=IntOptionHandler.class)
	private int outputs;
	@Option(name="-fte",usage="number of fine tune epochs to train on (default: 100)",handler=IntOptionHandler.class)
	private int finetuneEpochs = 100;
	@Option(name="-pte",usage="number of epochs for pretraining (default: 100)",handler=IntOptionHandler.class)
	private int pretrainEpochs = 100;
	@Option(name="-r",usage="seed value for the random number generator (default: 123)",handler=IntOptionHandler.class)
	private long rngSeed = 123;
	@Option(name="-k",usage="the k for rbms (default: 1)",handler=IntOptionHandler.class)
	private int k;
	@Option(name="-c",usage="corruption level (for denoising autoencoders) (default: 0.3)",handler=DoubleOptionHandler.class)
	private double corruptionLevel = 0.3;
	@Option(name="-h",usage="the host to connect to as a master (default: 127.0.0.1)")
	private String host = "127.0.0.1";
	@Option(name="-ftl",usage="the starter fine tune learning rate (default: 0.1)",handler=DoubleOptionHandler.class)
	private double finetineLearningRate = 0.1;
	@Option(name="-ptl",usage="the starter pretrain learning rate (default: 0.1)",handler=DoubleOptionHandler.class)
	private double pretrainLearningRate = 0.1;
	@Option(name="-hl",usage="hidden layer sizes (comma separated list)")
	private String hiddenLayerSizesOption;
	private int[] hiddenLayerSizes = {300,300,300};
	@Option(name="-t",usage="type of worker")
	private String type = "master";
	@Option(name="-ad",usage="address of master worker")
	private String address;
	@Option(name="-sp",usage="number of inputs to split by default: 10")
	private int split = 10;
	@Option(name="-data",usage="dataset to train on: options: mnist,text (text files with <label>text</label>, image (images where the parent directory is the label)")
	private String dataSet;
	@Option(name="-e",usage="number of examples to train on: if unspecified will just train on everything found")
	private int numExamples = -1;
	private ActorNetworkRunner runner;
	private DataSetIterator iter;


	public ActorNetworkRunnerApp(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
			ensureValidMinConf();
			getDataSet();
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


	private void ensureValidMinConf() {
		if(this.algorithm == null && !Arrays.asList("sda","cdbn","dbn").contains(this.algorithm))
			throw new IllegalStateException("Algorithm not defined or invalid algorithm specified");
		if(this.inputs < 1)
			throw new IllegalStateException("Please define some inputs");
		if(this.outputs < 1)
			throw new IllegalStateException("Please define some outputs");
		
	}


	public void exec() throws Exception {

		//this is just a worker node: load everything from the master. All we should need is the ip of the master
		//to set everything up
		if(type != null && type.equals("worker"))  {
			ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever(host, 2181, "master");
			Conf conf = retriever.retreive();
			String address = conf.get(MASTER_URL);
			runner = new ActorNetworkRunner(type,address);
			runner.setup(conf);
			retriever.close();

		}
		else {
			Conf conf = new Conf();



			conf.put(CLASS, getClassForAlgorithm());
			conf.put(LAYER_SIZES, Arrays.toString(hiddenLayerSizes).replace("[","").replace("]","").replace(" ",""));
			conf.put(SPLIT,String.valueOf(10));
			conf.put(N_IN, String.valueOf(iter.inputColumns()));
			conf.put(OUT, String.valueOf(iter.totalOutcomes()));
			conf.put(PRE_TRAIN_EPOCHS, String.valueOf(pretrainEpochs));
			conf.put(FINE_TUNE_EPOCHS, String.valueOf(finetuneEpochs));
			conf.put(SEED, String.valueOf(rngSeed));
			conf.put(LEARNING_RATE,String.valueOf(pretrainLearningRate));

			conf.put(LAYER_SIZES, Arrays.toString(hiddenLayerSizes).replace("[","").replace("]","").replace(" ",""));
			conf.put(CORRUPTION_LEVEL,corruptionLevel);
			conf.put(SPLIT, String.valueOf(split));
			conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(corruptionLevel).finetuneEpochs(finetuneEpochs)
					.finetuneLearningRate(finetineLearningRate).learningRate(pretrainLearningRate).epochs(pretrainEpochs).build());

			//run the master
			runner = new ActorNetworkRunner();
			runner.setup(conf);
			//store it in zookeeper for service discovery
			conf.put(MASTER_URL, runner.getMasterAddress().toString());

			//register the configuration to zookeeper
			ZooKeeperConfigurationRegister reg = new ZooKeeperConfigurationRegister(conf,"master","localhost",2181);
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
		
		Pair<DoubleMatrix,DoubleMatrix> batch = null;
		if(iter.hasNext()) {
			batch = iter.next();
			runner.train(batch);
		}
	}


	public void shutdown() {
		
	}

	public String getData() {
		return dataSet;
	}

	private void getDataSet() {
		try {
			if(dataSet.equals("mnist")) {
				iter = new MnistDataSetIterator(numExamples);
			}

			else if(dataSet.equals("iris")) {
				iter = new IrisDataSetIterator(numExamples);
			}
			else if(dataSet.equals("lfw")) {
				iter = new LFWDataSetIterator(numExamples);
			}

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}


	private String getClassForAlgorithm() {
		switch(algorithm) {
		case  "sda" :
			return "com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix";

		case "dbn" : 
			return "com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas.DBN";
		case "cdbn":
			return "com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas.CDBN";
		}
		return null;
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
		return finetineLearningRate;
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
