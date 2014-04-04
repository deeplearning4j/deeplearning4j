package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.DataOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.DoneMessage;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.actor.core.NoJobFound;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.PoisonPill;
import akka.actor.Props;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.routing.RoundRobinPool;


/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableImpl> {

	protected BaseMultiLayerNetwork network;
	protected AtomicLong lastUpdated = new AtomicLong(System.currentTimeMillis());
	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 * @param batchActor the batch actor that handles data set dispersion
	 */
	public MasterActor(Conf conf,ActorRef batchActor,HazelCastStateTracker stateTracker) {
		super(conf,batchActor,stateTracker);
		setup(conf);
		/*
		 * Ensures there's no one off errors by forcing the next phase as well as ensures
		 * that if the system is done it shuts down
		 */
		forceNextPhase =  context().system().scheduler()
				.schedule(Duration.create(1,TimeUnit.MINUTES), Duration.create(1,TimeUnit.MINUTES), new Runnable() {

					@Override
					public void run() {
						try {
							List<Job> currentJobs = MasterActor.this.stateTracker.currentJobs();
							log.info("Status check on next iteration");

							if(updates.size() >= partition || currentJobs.isEmpty())
								nextIteration();



							log.info("Current jobs left " + currentJobs);


						}catch(Exception e) {
							throw new RuntimeException(e);
						}

					}

				}, context().dispatcher());

		this.clearStateWorkers =  context().system().scheduler()
				.schedule(Duration.create(1,TimeUnit.MINUTES), Duration.create(1,TimeUnit.MINUTES), new Runnable() {

					@Override
					public void run() {
						try {
							long now = System.currentTimeMillis();
							Map<String,Long> heartbeats = MasterActor.this.stateTracker.getHeartBeats();
							for(String key : heartbeats.keySet()) {
								long lastChecked = heartbeats.get(key);
								long diff = now - lastChecked;
								long seconds = TimeUnit.MILLISECONDS.toSeconds(diff);
								if(seconds >= 30) {
									log.info("Removing stale worker " + key);
									MasterActor.this.stateTracker.removeWorker(key);
									partition--;
								}

							}



						}catch(Exception e) {
							throw new RuntimeException(e);
						}

					}

				}, context().dispatcher());

	}






	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 * @param batchActor the batch actor for the cluster, this
	 * will manage dataset dispersion
	 * @param network the neural network to use
	 */
	public MasterActor(Conf conf,ActorRef batchActor,BaseMultiLayerNetwork network,HazelCastStateTracker stateTracker) {
		super(conf,batchActor,stateTracker);
		this.network = network;
		setup(conf);

	}



	@Override
	public  UpdateableImpl compute(Collection<UpdateableImpl> workerUpdates,
			Collection<UpdateableImpl> masterUpdates) {


		DeepLearningAccumulator acc = new DeepLearningAccumulator();
		for(UpdateableImpl m : workerUpdates) 
			acc.accumulate(m.get());
		UpdateableImpl masterResults = this.getResults();
		if(masterResults == null)
			masterResults = new UpdateableImpl(acc.averaged());
		else
			masterResults.set(acc.averaged());

		try {
			stateTracker.setCurrent(masterResults);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}


		return masterResults;
	}



	@Override
	public void setup(Conf conf) {
		log.info("Starting workers");
		ActorSystem system = context().system();
		RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());
		//start local workers
		Props p = pool.props(WorkerActor.propsFor(conf,stateTracker));
		p = ClusterSingletonManager.defaultProps(p, "master", PoisonPill.getInstance(), "master");

		system.actorOf(p, "worker");





		//Wait for backend to be up

		try {
			Thread.sleep(30000);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}


		log.info("Broadcasting initial master network");

		BaseMultiLayerNetwork network = this.network == null ? new BaseMultiLayerNetwork.Builder<>()
				.numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
				.hiddenLayerSizes(conf.getLayerSizes()).renderWeights(conf.getRenderWeightEpochs())
				.useRegularization(conf.isUseRegularization()).withDropOut(conf.getDropOut()).withLossFunction(conf.getLossFunction())
				.withSparsity(conf.getSparsity()).useAdaGrad(conf.isUseAdaGrad()).withOptimizationAlgorithm(conf.getOptimizationAlgorithm())
				.withMultiLayerGradientListeners(conf.getMultiLayerGradientListeners())
				.withGradientListeners(conf.getGradientListeners())
				.build() : this.network;

				//ensure rng is synchronized whether its loaded from an external source or not
				network.synchonrizeRng();


				if(conf.getColumnMeans() != null)
					network.setColumnMeans(conf.getColumnMeans());
				if(conf.getColumnStds() != null)
					network.setColumnStds(conf.getColumnStds());

				UpdateableImpl masterResults = new UpdateableImpl(network);

				try {
					this.stateTracker.setCurrent(masterResults);
				} catch (Exception e1) {
					throw new RuntimeException(e1);
				}


				//after worker is instantiated broadcast the master network to the worker
				mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
						masterResults), getSelf());	

	}







	/**
	 * Check on the next iteration
	 * @throws Exception
	 */
	protected void nextIteration() throws Exception {
		
		if(!updates.isEmpty()) {
			UpdateableImpl masterResults = this.compute(updates, updates);


			epochsComplete++;
			//tell the batch actor to send out another dataset
			if(!isDone())
				batchActor.tell(new MoreWorkMessage(masterResults), getSelf());
			updates.clear();
			log.info("Broadcasting weights");
			//replicate the network
			mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
					masterResults), getSelf());
			this.stateTracker.setCurrent(masterResults);

		}
	
		checkDone();
	

	}

	/**
	 * Checks if done
	 * @throws Exception
	 */
	protected void checkDone() throws Exception {
		UpdateableImpl masterResults = null;
		if(!updates.isEmpty()) {
			masterResults = compute(updates, updates);

			stateTracker.setCurrent(masterResults);


			epochsComplete++;
			updates.clear();

		}

		else 
			masterResults = getMasterResults();

		if(stateTracker.isPretrain() && stateTracker.currentJobs().isEmpty()) {
			log.info("Switching to finetune mode");
			pretrain = false;
			stateTracker.moveToFinetune();
			SerializationUtils.saveObject(masterResults.get(), new File("pretrain-model.bin"));


			batchActor.tell(ResetMessage.getInstance(), getSelf());
			batchActor.tell(new MoreWorkMessage(masterResults), getSelf());

		}

		else if(stateTracker.currentJobs().isEmpty()) {
			isDone = true;
			stateTracker.finish();
			log.info("Done training!");
		}



	}


	@SuppressWarnings({ "unchecked" })
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck || message instanceof DistributedPubSubMediator.UnsubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			//reply
			mediator.tell(new DistributedPubSubMediator.Publish(ClusterListener.TOPICS,
					message), getSelf());	


			log.info("Subscribed " + ack.toString());
		}



		else if(message instanceof NoJobFound) {
			partition--;
			if(updates.size() >= partition) 
				nextIteration();

		}

		else if(message instanceof DoneMessage) {
			log.info("Received done message");
			checkDone();
		}


		else if(message instanceof String) {
			getSender().tell(Ack.getInstance(),getSelf());

		}


		else if(message instanceof UpdateableImpl) {
			UpdateableImpl up = (UpdateableImpl) message;
			updates.add(up);
			log.info("Num updates so far " + updates.size() + " and partition size is " + partition);

			//note that partition is always the current number of workers that was dispatched to
			//this means that the number of workers will never outpace the number of datasets
			if(updates.size() >= partition) 
				nextIteration();

		}


		//list of examples
		else if(message instanceof List || message instanceof DataSet) {

			if(message instanceof List) {
				List<DataSet> list = (List<DataSet>) message;
				//each pair in the matrix pairs maybe multiple rows
				splitListIntoRows(list);
				//delegate split to workers
				sendToWorkers(list);

			}

			//ensure split then send to workers
			else if(message instanceof DataSet) {
				DataSet pair = (DataSet) message;

				//split pair up in to rows to ensure parallelism
				List<DoubleMatrix> inputs = pair.getFirst().rowsAsList();
				List<DoubleMatrix> labels = pair.getSecond().rowsAsList();

				List<DataSet> pairs = new ArrayList<>();
				for(int i = 0; i < inputs.size(); i++) {
					pairs.add(new DataSet(inputs.get(i),labels.get(i)));
				}


				sendToWorkers(pairs);

			}
		}

		else
			unhandled(message);
	}


	@Override
	public void complete(DataOutputStream ds) {
		this.getMasterResults().get().write(ds);
	}



}
