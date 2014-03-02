package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.actor.core.FinetuneMessage;
import org.deeplearning4j.iterativereduce.actor.core.actor.BatchActor;
import org.deeplearning4j.iterativereduce.actor.core.actor.ModelSavingActor;
import org.deeplearning4j.iterativereduce.actor.core.actor.SimpleClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.api.EpochDoneListener;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.scaleout.zookeeper.ZooKeeperConfigurationRegister;
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
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.routing.RoundRobinPool;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;


/**
 * Controller for coordinating model training for a neural network based
 * on parameters across a cluster for akka.
 * @author Adam Gibson
 *
 */
public class ActorNetworkRunner implements DeepLearningConfigurable,EpochDoneListener<UpdateableImpl> {


	private static final long serialVersionUID = -4385335922485305364L;
	private transient ActorSystem system;
	private Integer currEpochs = 0;
	private Integer epochs;
	private UpdateableImpl result;
	private ActorRef mediator;
	private static Logger log = LoggerFactory.getLogger(ActorNetworkRunner.class);
	private static String systemName = "ClusterSystem";
	private String type = "master";
	private Address masterAddress;
	private DataSetIterator iter;

	/**
	 * Master constructor
	 * @param type the type (worker)
	 * @param iter the dataset to use
	 */
	public ActorNetworkRunner(String type,DataSetIterator iter) {
		this.type = type;
		this.iter = iter;
	}

	/**
	 * The worker constructor
	 * @param type the type to use
	 * @param address the address of the master
	 */
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
	public static Address startBackend(Address joinAddress, String role,Conf c,DataSetIterator iter) {
		Config conf = ConfigFactory.parseString("akka.cluster.roles=[" + role + "]").
				withFallback(ConfigFactory.load());
		ActorSystem system = ActorSystem.create(systemName, conf);
		
		RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());

		
		ActorRef batchActor = system.actorOf(pool.props(Props.create(new BatchActor.BatchActorFactory(iter,c.getNumPasses()))));
		
		log.info("Started batch actor");
		
		
		/*
		 * Starts a master: in the active state with the poison pill upon failure with the role of master
		 */
		Address realJoinAddress =
				(joinAddress == null) ? Cluster.get(system).selfAddress() : joinAddress;
				Cluster.get(system).join(realJoinAddress);
				system.actorOf(ClusterSingletonManager.defaultProps(MasterActor.propsFor(c,batchActor), "active", PoisonPill.getInstance(), "master"));
		return realJoinAddress;
	}


	public static ActorRef startWorker(Address contactAddress,Conf conf) {
		// Override the configuration of the port

		ActorSystem system = ActorSystem.create(systemName);
		Set<ActorSelection> initialContacts = new HashSet<ActorSelection>();

		initialContacts.add(system.actorSelection(contactAddress + "/user/receptionist"));
		ActorRef clusterClient = system.actorOf(ClusterClient.defaultProps(initialContacts),
				"clusterClient");
		Props p = WorkerActor.propsFor(clusterClient, conf);
		int cores = Runtime.getRuntime().availableProcessors();
		
		ActorRef ref = system.actorOf(new RoundRobinPool(cores).props(p), "worker");
		
		log.info("Worker joining cluster");
		Cluster.get(system).join(contactAddress);
		
		
		return ref;
	}



	@Override
	public void setup(final Conf conf) {



		system = ActorSystem.create(systemName);

		// Create an actor that handles cluster domain events
		system.actorOf(Props.create(SimpleClusterListener.class),
				"clusterListener");


		mediator = DistributedPubSubExtension.get(system).mediator();

		epochs = conf.getPretrainEpochs();
		if(type.equals("master")) {

			if(iter == null)
				throw new IllegalStateException("Unable to initialize no dataset to train");
			
			log.info("Starting master");
			masterAddress  = startBackend(null,"master",conf,iter);
			
			
			try {
				Thread.sleep(30000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
					this), mediator);
			
			log.info("Starting model saver");
			ActorRef ref = system.actorOf(Props.create(ModelSavingActor.class,",model-saver"));
			
		
			//MAKE SURE THIS ACTOR SYSTEM JOINS THE CLUSTER;
			//There is a one to one join to system requirement for the cluster
			Cluster.get(system).join(masterAddress);

		
			
			
		
			log.info("Setup master with epochs " + epochs);
		}

		else {

			Conf c = conf.copy();
	


			ActorRef worker = startWorker(masterAddress,c);

			//Wait for backend to be up

			try {
				Thread.sleep(30000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}

		
			
			//MAKE SURE THIS ACTOR SYSTEM JOINS THE CLUSTER;
			//There is a one to one join to system requirement for the cluster
			Cluster.get(system).join(masterAddress);


			log.info("Setup worker nodes");
		}
		
		//store it in zookeeper for service discovery
		conf.setMasterUrl(getMasterAddress().toString());

		scala.concurrent.Future<Void> f = Futures.future(new Callable<Void>() {

			@Override
			public Void call() throws Exception {
				log.info("Registering with zookeeper; if the logging stops here, ensure zookeeper is started");
				//register the configuration to zookeeper
				ZooKeeperConfigurationRegister reg = new ZooKeeperConfigurationRegister(conf,"master","localhost",2181);
				reg.register();
				reg.close();
				return null;
			}
			
		},system.dispatcher());
		
		f.onComplete(new OnComplete<Void>() {

			@Override
			public void onComplete(Throwable arg0, Void arg1) throws Throwable {
				if(arg0 != null)
					throw arg0;
				log.info("Registered conf with zookeeper");

			}
			
		}, system.dispatcher());
		
		
	}

	

	public void train(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
		log.info("Publishing to results for training");
		//wait for cluster to be up
		try {
			log.info("Waiting for cluster to go up...");
			Thread.sleep(30000);
			log.info("Done waiting");
		} catch (InterruptedException e1) {
			Thread.currentThread().interrupt();
		}

		//ensure the trainer is known so the next iteration can happen
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				this), mediator);
		log.info("Started pipeline");
		//start the pipeline
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				list), mediator);

		log.info("Published results");
	}



	public void train() {
		if(iter.hasNext())
			train(iter.next());
		
		else
			log.warn("No data found");
	}

	public void train(Pair<DoubleMatrix,DoubleMatrix> input) {
		train(new ArrayList<>(Arrays.asList(input)));

	}



	public void train(DoubleMatrix input,DoubleMatrix labels) {
		train(new Pair<DoubleMatrix,DoubleMatrix>(input,labels));
	}


	@Override
	public void epochComplete(UpdateableImpl result) {
		currEpochs++;

		//update the final available result
		this.result = result;
		mediator.tell(new DistributedPubSubMediator.Publish(BatchActor.FINETUNE,
				new FinetuneMessage(result)), mediator);
		

	}

	public UpdateableImpl getResult() {
		return result;
	}



	@Override
	public void finish() {

	}

	public Address getMasterAddress() {
		return masterAddress;
	}



}
