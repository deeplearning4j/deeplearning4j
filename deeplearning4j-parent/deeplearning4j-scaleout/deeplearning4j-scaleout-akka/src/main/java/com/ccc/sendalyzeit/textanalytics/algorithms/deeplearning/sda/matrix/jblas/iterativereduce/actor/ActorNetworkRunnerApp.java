package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.util.Arrays;

import org.apache.commons.lang3.StringUtils;
import org.apache.zookeeper.ZooKeeper;
import org.jblas.DoubleMatrix;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.DoubleOptionHandler;
import org.kohsuke.args4j.spi.IntOptionHandler;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZooKeeperConfigurationRegister;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZookeeperBuilder;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZookeeperConfigurationRetriever;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurableDistributed;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.ExtraParamsBuilder;

public class ActorNetworkRunnerApp implements DeepLearningConfigurableDistributed {


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
	private String type;
	@Option(name="-ad",usage="address of master worker")
	private String address;
	@Option(name="-sp",usage="number of inputs to split by default: 10")
	private int split = 10;

	public ActorNetworkRunnerApp() {
	}


	public void exec(String[] args) throws Exception {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
			if(hiddenLayerSizesOption != null) {
				String[] split = hiddenLayerSizesOption.split(",");
				hiddenLayerSizes = new int[split.length];
				for(int i = 0; i < split.length; i++) {
					hiddenLayerSizes[i] = Integer.parseInt(split[i]);
				}


			}
			
			//this is just a worker node: load everything from the master. All we should need is the ip of the master
			//to set everything up
			if(type != null && type.equals("worker"))  {
				ZooKeeper zk = new ZookeeperBuilder().setHost(host).build();
				ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever(zk, "master");
				Conf conf = retriever.retreive();
				String address = conf.get(MASTER_URL);
				ActorNetworkRunner runner = new ActorNetworkRunner(type,address);
				runner.setup(conf);

			}
			else {
				Conf conf = new Conf();

				Pair<DoubleMatrix,DoubleMatrix> mnist = DeepLearningTest.getMnistExampleBatch(100);

				conf.put(CLASS, getClassForAlgorithm());
				conf.put(LAYER_SIZES, Arrays.toString(hiddenLayerSizes).replace("[","").replace("]","").replace(" ",""));
				conf.put(SPLIT,String.valueOf(10));
				conf.put(N_IN, String.valueOf(mnist.getFirst().columns));
				conf.put(OUT, String.valueOf(mnist.getSecond().columns));
				conf.put(PRE_TRAIN_EPOCHS, String.valueOf(pretrainEpochs));
				conf.put(FINE_TUNE_EPOCHS, String.valueOf(finetuneEpochs));
				conf.put(SEED, String.valueOf(rngSeed));
				conf.put(LEARNING_RATE,String.valueOf(pretrainLearningRate));

				conf.put(LAYER_SIZES, StringUtils.join(hiddenLayerSizes,","));
				conf.put(CORRUPTION_LEVEL,corruptionLevel);
				conf.put(SPLIT, String.valueOf(split));
				conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(corruptionLevel).finetuneEpochs(finetuneEpochs)
						.finetuneLearningRate(finetineLearningRate).learningRate(pretrainLearningRate).epochs(pretrainEpochs).build());

				//run the master
				ActorNetworkRunner runner = new ActorNetworkRunner();
				runner.setup(conf);
				//store it in zookeeper for service discovery
				conf.put(MASTER_URL, runner.getMasterAddress().toString());
				
				//register the configuration to zookeeper
				ZooKeeperConfigurationRegister reg = new ZooKeeperConfigurationRegister(conf,new ZookeeperBuilder().build(),"master");
				reg.register();
				
				
				
			}


		} catch (Exception e) {
			parser.printUsage(System.err);
			System.exit(1);
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
		new ActorNetworkRunnerApp().exec(args);
	}

	@Override
	public void setup(Conf conf) {

	}

}
