package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.Serializable;
import java.net.URI;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.actor.BatchActor;
import org.deeplearning4j.iterativereduce.actor.core.actor.ModelSavingActor;
import org.deeplearning4j.iterativereduce.actor.util.ActorRefUtils;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.ActorSelection;
import akka.actor.ActorSystem;
import akka.actor.Address;
import akka.actor.AddressFromURIString;
import akka.actor.PoisonPill;
import akka.actor.Props;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterClient;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.routing.RoundRobinPool;
/**
 * Controller for coordinating model training for a neural network based
 * on parameters across a cluster for akka.
 * @author Adam Gibson
 *
 */
public class ActorNetworkRunner implements DeepLearningConfigurable,Serializable {


	private static final long serialVersionUID = -4385335922485305364L;
	private transient ActorSystem system;
	private Integer epochs;
	private ActorRef mediator;
	private BaseMultiLayerNetwork startingNetwork;
	private static Logger log = LoggerFactory.getLogger(ActorNetworkRunner.class);
	private static String systemName = "ClusterSystem";
	private String type = "master";
	private Address masterAddress;
	private DataSetIterator iter;
	protected ActorRef masterActor;
	private transient ScheduledExecutorService exec;
	private transient StateTracker<UpdateableImpl> stateTracker;
	private Conf conf;
	private boolean finetune = false;

	/**
	 * Master constructor
	 * @param type the type (worker)
	 * @param iter the dataset to use
	 * @param startingNetwork a starting neural network
	 */
	public ActorNetworkRunner(String type,DataSetIterator iter,BaseMultiLayerNetwork startingNetwork) {
		this.type = type;
		this.iter = iter;
		this.startingNetwork = startingNetwork;
	}

	/**
	 * Master constructor
	 * @param type the type (worker)
	 * @param iter the dataset to use
	 * @param startingNetwork a starting neural network
	 */
	public ActorNetworkRunner(String type,DataSetIterator iter) {
		this(type,iter,null);
	}


	/**
	 * Master constructor
	 * @param type the type (worker)
	 * @param iter the dataset to use
	 */
	public ActorNetworkRunner(DataSetIterator iter) {
		this("master",iter,null);
	}


	/**
	 * Master constructor
	 * @param type the type (worker)
	 * @param iter the dataset to use
	 */
	public ActorNetworkRunner(DataSetIterator iter,BaseMultiLayerNetwork startingNetwork) {
		this("master",iter,startingNetwork);
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
	public Address startBackend(Address joinAddress, String role,Conf c,DataSetIterator iter,StateTracker<UpdateableImpl> stateTracker) {

		ActorRefUtils.addShutDownForSystem(system);

		system.actorOf(Props.create(ClusterListener.class));

		ActorRef batchActor = system.actorOf(Props.create(BatchActor.class,iter,stateTracker,c),"batch");

		log.info("Started batch actor");

		Props masterProps = startingNetwork != null ? Props.create(MasterActor.class,c,batchActor,startingNetwork,stateTracker) : Props.create(MasterActor.class,c,batchActor,stateTracker);

		/*
		 * Starts a master: in the active state with the poison pill upon failure with the role of master
		 */
		final Address realJoinAddress = (joinAddress == null) ? Cluster.get(system).selfAddress() : joinAddress;


		c.setMasterUrl(realJoinAddress.toString());

		if(exec == null)
			exec = Executors.newScheduledThreadPool(2);


		Cluster cluster = Cluster.get(system);
		cluster.join(realJoinAddress);

		exec.schedule(new Runnable() {

			@Override
			public void run() {
				Cluster cluster = Cluster.get(system);
				cluster.publishCurrentClusterState();
			}

		}, 10, TimeUnit.SECONDS);

		masterActor = system.actorOf(ClusterSingletonManager.defaultProps(masterProps, "master", PoisonPill.getInstance(), "master"));

		log.info("Started master with address " + realJoinAddress.toString());
		c.setMasterAbsPath(ActorRefUtils.absPath(masterActor, system));
		log.info("Set master abs path " + c.getMasterAbsPath());

		return realJoinAddress;
	}


	/**
	 * Automatically switch to finetune step
	 */
	public void finetune() {
		this.finetune = true;
		if(this.startingNetwork == null) {
			throw new IllegalStateException("No network to finetune!");
		}
	}
	
	
	@Override
	public void setup(final Conf conf) {



		system = ActorSystem.create(systemName);
		ActorRefUtils.addShutDownForSystem(system);
		mediator = DistributedPubSubExtension.get(system).mediator();

		epochs = conf.getPretrainEpochs();
		if(type.equals("master")) {

			if(iter == null)
				throw new IllegalStateException("Unable to initialize no dataset to train");

			log.info("Starting master");

			try {
				stateTracker = new HazelCastStateTracker();
				if(finetune)
					stateTracker.moveToFinetune();
				
				masterAddress  = startBackend(null,"master",conf,iter,stateTracker);
				Thread.sleep(60000);

			} catch (Exception e1) {
				Thread.currentThread().interrupt();
				throw new RuntimeException(e1);
			}




			log.info("Starting model saver");
			system.actorOf(Props.create(ModelSavingActor.class,"model-saver"));


			//store it in zookeeper for service discovery
			conf.setMasterUrl(getMasterAddress().toString());
			conf.setMasterAbsPath(ActorRefUtils.absPath(masterActor, system));

			ActorRefUtils.registerConfWithZooKeeper(conf, system);


			system.scheduler().schedule(Duration.create(1, TimeUnit.MINUTES), Duration.create(1, TimeUnit.MINUTES), new Runnable() {

				@Override
				public void run() {
					log.info("Current cluster members " + Cluster.get(system).readView().members());
				}

			},system.dispatcher());
			log.info("Setup master with epochs " + epochs);
		}

		else {

			Address a = AddressFromURIString.parse(conf.getMasterUrl());
			
			Conf c = conf.copy();
			Cluster cluster = Cluster.get(system);
			cluster.join(a);

			try {
				String host = a.host().get();
				
				if(host == null)
					throw new IllegalArgumentException("No host set for worker");
				
				int port = HazelCastStateTracker.DEFAULT_HAZELCAST_PORT;
				
				String connectionString = host + ":" + port;
				
				stateTracker = new HazelCastStateTracker(connectionString,"worker");

			} catch (Exception e1) {
				Thread.currentThread().interrupt();
				throw new RuntimeException(e1);
			}
			
			startWorker(c);

			system.scheduler().schedule(Duration.create(1, TimeUnit.MINUTES), Duration.create(1, TimeUnit.MINUTES), new Runnable() {

				@Override
				public void run() {
					log.info("Current cluster members " + Cluster.get(system).readView().members());
				}

			},system.dispatcher());
			log.info("Setup worker nodes");
		}

		this.conf = conf;

	}


	public  void startWorker(Conf conf) {
		
		Address contactAddress = AddressFromURIString.parse(conf.getMasterUrl());

		system.actorOf(Props.create(ClusterListener.class));
		log.info("Attempting to join node " + contactAddress);
		log.info("Starting workers");
		Set<ActorSelection> initialContacts = new HashSet<ActorSelection>();
		initialContacts.add(system.actorSelection(contactAddress + "/user/"));

		RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());

		ActorRef clusterClient = system.actorOf(ClusterClient.defaultProps(initialContacts),
				"clusterClient");


		try {
			String host = contactAddress.host().get();
			log.info("Connecting hazelcast to host " + host);
			int workers = stateTracker.numWorkers();
			if(workers <= 1)
				throw new IllegalStateException("Did not properly connect to cluster");
			
			
			log.info("Joining cluster of size " + workers);


			Props p = pool.props(WorkerActor.propsFor(clusterClient,conf,stateTracker));
			system.actorOf(p, "worker");

			Cluster cluster = Cluster.get(system);
			cluster.join(contactAddress);

			log.info("Worker joining cluster");


		} catch (Exception e) {
			throw new RuntimeException(e);
		}




	}


	public void train(List<DataSet> list) {
		log.info("Publishing to results for training");
		//wait for cluster to be up
		try {
			log.info("Waiting for cluster to go up...");
			Thread.sleep(30000);
			log.info("Done waiting");
		} catch (InterruptedException e1) {
			Thread.currentThread().interrupt();
		}


		log.info("Started pipeline");
		//start the pipeline
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				list), mediator);

		log.info("Published results");
	}


	/**
	 * Kicks off the distributed training.
	 * It will grab the optimal batch size off of 
	 * the beginning of the dataset iterator which
	 * is based on the desired mini batch size (conf.getSplit())
	 * 
	 * and the number of initial workers in the state tracker after setup.
	 * 
	 * For example, if you have a mini batch size of 10 and 8 workers
	 * 
	 * the initial @link{DataSetIterator#next(int batches)} would be 
	 * 
	 * 80, this would be 10 per worker.
	 */
	public void train() {
		int numWorkers = stateTracker.numWorkers();
		int batch = conf.getSplit();
		int miniBatches = numWorkers * batch;
		if(iter.hasNext())
			train(iter.next(miniBatches));

		else
			log.warn("No data found");
	}

	public void train(DataSet input) {
		List<DataSet> list = input.asList();
		train(list);

	}



	public void train(DoubleMatrix input,DoubleMatrix labels) {
		train(new DataSet(input,labels));
	}




	public Address getMasterAddress() {
		return masterAddress;
	}

	public synchronized StateTracker<UpdateableImpl> getStateTracker() {
		return stateTracker;
	}

	public synchronized void setStateTracker(
			StateTracker<UpdateableImpl> stateTracker) {
		this.stateTracker = stateTracker;
	}

	public void shutdown() {
		if(stateTracker != null)
			stateTracker.shutdown();
		system.shutdown();
	}



}
