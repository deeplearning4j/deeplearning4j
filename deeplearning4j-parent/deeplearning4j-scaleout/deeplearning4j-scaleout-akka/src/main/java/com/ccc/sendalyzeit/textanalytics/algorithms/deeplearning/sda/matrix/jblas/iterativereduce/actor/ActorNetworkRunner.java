package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.ActorSelection;
import akka.actor.ActorSystem;
import akka.actor.Address;
import akka.actor.PoisonPill;
import akka.actor.Props;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterClient;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.deeplearning.eval.Evaluation;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurable;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.ExtraParamsBuilder;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;


/**
 * Controller for coordinating model training for a neural network based
 * on parameters across a cluster for akka.
 * @author Adam Gibson
 *
 */
public class ActorNetworkRunner implements DeepLearningConfigurable,EpochDoneListener {

	
	private static final long serialVersionUID = -4385335922485305364L;
	private transient ActorSystem system;
	private Integer currEpochs = 0;
	private Integer epochs;
	private List<Pair<DoubleMatrix,DoubleMatrix>> samples;
	private UpdateableMatrix result;
	private  ActorRef mediator;

	private static Logger log = LoggerFactory.getLogger(ActorNetworkRunner.class);
	private static String systemName = "ClusterSystem";
	private String type = "master";
	private Address masterAddress;

	public ActorNetworkRunner(String type) {
		this.type = type;
	}

	public ActorNetworkRunner(String type,String address) {
		this.type = type;
		URI u = URI.create(address);
		masterAddress = Address.apply(u.getScheme(), u.getUserInfo(), u.getHost(), u.getPort());
	}




	public ActorNetworkRunner() {
		super();
	}




	/**
	 * Start a backend with the given role
	 * @param joinAddress the join address
	 * @param role the role to start with
	 * @param c the neural network configuration
	 * @return the actor for this backend
	 */
	public static Address startBackend(Address joinAddress, String role,Conf c) {
		Config conf = ConfigFactory.parseString("akka.cluster.roles=[" + role + "]").
				withFallback(ConfigFactory.load());
		ActorSystem system = ActorSystem.create(systemName, conf);
		/*
		 * Starts a master: in the active state with the poison pill upon failure with the role of master
		 */
		Address realJoinAddress =
				(joinAddress == null) ? Cluster.get(system).selfAddress() : joinAddress;
				Cluster.get(system).join(realJoinAddress);
				system.actorOf(ClusterSingletonManager.defaultProps(MasterActor.propsFor(c), "active", PoisonPill.getInstance(), "master"));
				return realJoinAddress;
	}


	public static void startWorker(Address contactAddress,Conf conf) {
		// Override the configuration of the port

		ActorSystem system = ActorSystem.create(systemName);
		Set<ActorSelection> initialContacts = new HashSet<ActorSelection>();

		initialContacts.add(system.actorSelection(contactAddress + "/user/receptionist"));
		ActorRef clusterClient = system.actorOf(ClusterClient.defaultProps(initialContacts),
				"clusterClient");
		system.actorOf(WorkerActor.propsFor(clusterClient, conf), "worker");

		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		Cluster.get(system).join(contactAddress);
	}



	@Override
	public void setup(Conf conf) {



		system = ActorSystem.create(systemName);

		// Create an actor that handles cluster domain events
		system.actorOf(Props.create(SimpleClusterListener.class),
				"clusterListener");


		mediator = DistributedPubSubExtension.get(system).mediator();

		epochs = conf.getInt(PRE_TRAIN_EPOCHS);

		if(type.equals("master")) {

			masterAddress  = startBackend(null,"master",conf);
			//wait for start
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			
			system.actorOf(Props.create(new ModelSavingActor.ModelSavingActorFactory("nn-model.bin")),",model-saver");
			
			//MAKE SURE THIS ACTOR SYSTEM JOINS THE CLUSTER;
			//There is a one to one join to system requirement for the cluster
			Cluster.get(system).join(masterAddress);

			//join with itself
			startBackend(masterAddress,"master",conf);


			Conf c = conf.copy();
			//only one iteration per worker; this events out to number of epochs iterated
			//TODO: make this tunable
			c.put(FINE_TUNE_EPOCHS, 1);
			c.put(PRE_TRAIN_EPOCHS,1);
			
			//Wait for backend to be up
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			
			startWorker(masterAddress,c);
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
					this), mediator);
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
					epochs), mediator);
			log.info("Setup master with epochs " + epochs);
		}

		else {

			Conf c = conf.copy();
			//only one iteration per worker; this events out to number of epochs iterated
			//TODO: make this tunable
			c.put(FINE_TUNE_EPOCHS, 1);
			c.put(PRE_TRAIN_EPOCHS,1);
			
			
			startWorker(masterAddress,c);
			//Wait for backend to be up
			
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			
			//MAKE SURE THIS ACTOR SYSTEM JOINS THE CLUSTER;
			//There is a one to one join to system requirement for the cluster
			Cluster.get(system).join(masterAddress);

			
			log.info("Setup worker nodes");
		}
	}

	public void train(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
		this.samples = list;
		log.info("Publishing to results for training");
		//wait for cluster to be up
		try {
			Thread.sleep(15000);
		} catch (InterruptedException e1) {
			Thread.currentThread().interrupt();
		}
		
		//ensure the trainer is known so the next iteration can happen
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
				this), mediator);
		//start the pipeline
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
				list), mediator);
		
		//serializable way of handing convergence
		do {
			try {
				Thread.sleep(10000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		}while(currEpochs < epochs);
	}




	public void train(Pair<DoubleMatrix,DoubleMatrix> input) {
		train(new ArrayList<>(Arrays.asList(input)));

	}



	public void train(DoubleMatrix input,DoubleMatrix labels) {
		train(new Pair<DoubleMatrix,DoubleMatrix>(input,labels));
	}

	public void shutdown() {
		system.shutdown();
	}

	@Override
	public void epochComplete(UpdateableMatrix result) {
 		currEpochs++;
		if(currEpochs < epochs) {
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.BROADCAST,
					result), mediator);

			log.info("Updating result on epoch " + currEpochs);
			//This needs to happen to wait for state to propagate.
			try {
				Thread.sleep(15000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			log.info("Starting next epoch");

			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
					samples), mediator);

		}
		
		//update the final available result
		this.result = result;
		mediator.tell(new DistributedPubSubMediator.Publish(ModelSavingActor.SAVE,
				result), mediator);
		
	}

	public UpdateableMatrix getResult() {
		return result;
	}


	public static void main(String[] args) throws IOException, InterruptedException {
		//initialize with a type and uri
		final ActorNetworkRunner runner  = args.length >= 2 ? args.length >=3 ? new ActorNetworkRunner(args[1],args[2]) : new ActorNetworkRunner(args[0]) : new ActorNetworkRunner();
		Conf conf = new Conf();
		Integer[] hidden_layer_sizes_arr = {300, 300,300};
		double pretrain_lr = 0.1;


		int pretraining_epochs = 1000;
		double finetune_lr = 0.1;
		int finetune_epochs = 500;


		final Pair<DoubleMatrix,DoubleMatrix> mnist = DeepLearningTest.getMnistExampleBatch(100);

		conf.put(CLASS, "com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix");
		conf.put(LAYER_SIZES, Arrays.toString(hidden_layer_sizes_arr).replace("[","").replace("]","").replace(" ",""));
		conf.put(SPLIT,String.valueOf(10));
		conf.put(N_IN, String.valueOf(mnist.getFirst().columns));
		conf.put(OUT, String.valueOf(mnist.getSecond().columns));
		conf.put(PRE_TRAIN_EPOCHS, 100);
		conf.put(FINE_TUNE_EPOCHS, 100);
		conf.put(SEED, 1);
		conf.put(LEARNING_RATE, 0.1);

		conf.put(LAYER_SIZES, StringUtils.join(hidden_layer_sizes_arr,","));
		conf.put(CORRUPTION_LEVEL, 0.3);
		conf.put(SPLIT, 1);
		conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(0.5).finetuneEpochs(finetune_epochs)
				.finetuneLearningRate(finetune_lr).learningRate(pretrain_lr).epochs(pretraining_epochs).build());

		runner.setup(conf);
		log.info("Running training");
		runner.train(mnist.getFirst(), mnist.getSecond());




		BaseMultiLayerNetwork trained = runner.getResult().get();
		Evaluation eval = new Evaluation();
		DoubleMatrix predicted = trained.predict(mnist.getFirst());
		eval.eval(mnist.getSecond(), predicted);
		log.info(eval.stats());
		runner.shutdown();	

	}

	@Override
	public void finish() {
		// TODO Auto-generated method stub
		
	}

}
