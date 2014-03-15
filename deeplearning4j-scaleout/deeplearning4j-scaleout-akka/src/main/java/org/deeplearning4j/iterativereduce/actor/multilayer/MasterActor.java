package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.DataOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.AlreadyWorking;
import org.deeplearning4j.iterativereduce.actor.core.ClearWorker;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.DoneMessage;
import org.deeplearning4j.iterativereduce.actor.core.GiveMeMyJob;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.actor.core.NeedsModelMessage;
import org.deeplearning4j.iterativereduce.actor.core.NeedsStatus;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.actor.util.ActorRefUtils;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Cancellable;
import akka.actor.PoisonPill;
import akka.actor.Props;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.routing.RoundRobinPool;


/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableImpl> {

	protected BaseMultiLayerNetwork network;
	protected Cancellable ensureNoLeftOvers,ensureDistribution;
	protected AtomicLong lastUpdated = new AtomicLong(System.currentTimeMillis());
	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 * @param batchActor the batch actor that handles data set dispersion
	 */
	public MasterActor(Conf conf,ActorRef batchActor) {
		super(conf,batchActor);

	}


	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 * @param batchActor the batch actor that handles data set dispersion
	 */
	public static Props propsFor(Conf conf,ActorRef batchActor) {
		return Props.create(MasterActor.class,conf,batchActor);
	}



	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 * @param batchActor the batch actor for the cluster, this
	 * will manage dataset dispersion
	 * @param network the neural network to use
	 */
	public MasterActor(Conf conf,ActorRef batchActor,BaseMultiLayerNetwork network) {
		super(conf,batchActor,new Object[]{network});

	}

	public static Props propsFor(Conf conf,ActorRef batchActor,BaseMultiLayerNetwork network) {
		return Props.create(MasterActor.class,conf,batchActor,network);
	}


	@Override
	public synchronized UpdateableImpl compute(Collection<UpdateableImpl> workerUpdates,
			Collection<UpdateableImpl> masterUpdates) {


		DeepLearningAccumulator acc = new DeepLearningAccumulator();
		for(UpdateableImpl m : workerUpdates) 
			acc.accumulate(m.get());

		if(masterResults == null)
			masterResults = new UpdateableImpl(acc.averaged());
		else
			masterResults.set(acc.averaged());

		return masterResults;
	}



	@Override
	public void setup(Conf conf) {
		log.info("Starting workers");
		ActorSystem system = context().system();
		RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());
		//start local workers
		Props p = pool.props(WorkerActor.propsFor(conf));
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
				.hiddenLayerSizes(conf.getLayerSizes()).renderWeights(conf.getRenderWeightEpochs()).useRegularization(conf.isUseRegularization())
				.withSparsity(conf.getSparsity()).useAdGrad(conf.isUseAdaGrad())
				.withMultiLayerGradientListeners(conf.getMultiLayerGradientListeners())
				.withGradientListeners(conf.getGradientListeners())
				.build() : this.network;
				if(conf.getColumnMeans() != null)
					network.setColumnMeans(conf.getColumnMeans());
				if(conf.getColumnStds() != null)
					network.setColumnStds(conf.getColumnStds());

				masterResults = new UpdateableImpl(network);

				//after worker is instantiated broadcast the master network to the worker
				mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
						masterResults), getSelf());	
				//every minute check the batch and if after one minute there are no more updates
				//clear them out and recirculate
				ensureNoLeftOvers = context().system().scheduler()
						.schedule(Duration.create(1,TimeUnit.MINUTES), Duration.create(1,TimeUnit.MINUTES), new Runnable() {

							@Override
							public void run() {
								log.info("Status check on next iteration");
								if(!updates.isEmpty() && currentJobs.isEmpty() || everyWorkerAvailable()) {
									log.info("Forcing next iteration");
									nextIteration();
								}

								else {
									for(Job j : currentJobs.values()) {
										mediator.tell(new DistributedPubSubMediator.Publish(j.getWorkerId(),
												NeedsStatus.getInstance()), getSelf());
									}
								}

								log.info("Available workers " + availableWorkers);


								log.info("Current jobs left " + currentJobs.keySet());


							}

						}, context().dispatcher());

				ensureDistribution = context().system().scheduler()
						.schedule(Duration.create(1,TimeUnit.MINUTES), Duration.create(1,TimeUnit.MINUTES), new Runnable() {

							@Override
							public void run() {
								for(final Job j : needsToBeRedistributed) {
									scala.concurrent.Future<Void> f = Futures.future(new Callable<Void>() {

										@Override
										public Void call() throws Exception {
											WorkerState state = workers.get(j.getWorkerId());
											if(state != null) {
												state.setAvailable(true);
												log.info("Worker " + state.getWorkerId() + " unavailable for work");
												if(j != null) {
													log.info("Redispatching work for id " + j.getWorkerId());
													scala.concurrent.Future<WorkerState> f = Futures.future(new Callable<WorkerState>() {

														@Override
														public WorkerState call() throws Exception {
															log.info("Fetching new worker...");
															WorkerState worker = nextAvailableWorker();
															return worker;
														}



													},context().dispatcher());


													f.onComplete(new OnComplete<WorkerState>() {

														@Override
														public void onComplete(Throwable arg0,
																WorkerState worker) throws Throwable {
															if(arg0 != null)
																throw arg0;
															log.info("Found new worker ");
															j.setWorkerId(worker.getWorkerId());									

															//replicate the network
															mediator.tell(new DistributedPubSubMediator.Publish(worker.getWorkerId(),
																	j), getSelf());
															log.info("Delegated work to worker " + worker.getWorkerId());

														}

													}, context().dispatcher());

												}

											}					
											return null;
										}

									}, context().dispatcher());

									ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());
								}

							}

						}, context().dispatcher());
	}



	protected boolean everyWorkerAvailable() {
		for(WorkerState s : workers.values())
			if(!s.isAvailable())
				return false;
		return true;
	}

	protected void setWorkerDone(String id) {
		if(this.workers == null || id == null || !this.workers.containsKey(id))
			return;

		WorkerState state = this.workers != null ? this.workers.get(id) : null;
		if(state != null)
			this.workers.get(id).setAvailable(true);
		if(this.currentJobs != null && this.currentJobs.containsKey(id))
			this.currentJobs.remove(id);
		state.setAvailable(true);
		this.availableWorkers.add(state);
	}


	@Override
	public void postStop() throws Exception {
		super.postStop();
		ensureNoLeftOvers.cancel();
	}


	protected void nextIteration() {
		masterResults = this.compute(updates, updates);
		for(String key : workers.keySet()) {
			workers.get(key).setAvailable(true);
			currentJobs.remove(key);
			this.availableWorkers.add(workers.get(key));
			log.info("Freeing " + key + " for work post batch completion");
		}

		epochsComplete++;
		//tell the batch actor to send out another dataset
		if(!isDone())
			batchActor.tell(new MoreWorkMessage(masterResults), getSelf());
		updates.clear();
		log.info("Broadcasting weights");
		//replicate the network
		mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
				masterResults), getSelf());
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


		else if(message instanceof WorkerState) {
			WorkerState s = (WorkerState) message;
			if(s.getWorkerId() != null) {
				this.addWorker(s);
				getSender().tell(Ack.getInstance(),getSelf());

			}


		}



		else if(message instanceof GiveMeMyJob) {
			GiveMeMyJob j = (GiveMeMyJob) message;
			Job j2 = this.currentJobs.get(j.getId());
			if(j2 == null) {
				log.info("There does not appear to be a job for worker " + j.getId());
			}
			else {
				j.setJob(j2);

				log.info("Returning current job for worker " + j.getId());

				mediator.tell(new DistributedPubSubMediator.Publish(j.getId(),
						j), getSelf());	

			}


		}

		else if(message instanceof NeedsModelMessage) {
			log.info("Sending networks over");
			getSender().tell(masterResults.get(),getSelf());
		}

		else if(message instanceof DoneMessage) {
			log.info("Received done message");
			masterResults = this.compute(updates, updates);

			epochsComplete++;
			updates.clear();


			if(pretrain) {
				batchActor.tell(ResetMessage.getInstance(), getSelf());
				log.info("Switching to finetune mode");
				pretrain = false;
				SerializationUtils.saveObject(masterResults.get(), new File("pretrain-model.bin"));

				for(String key : workers.keySet()) {
					setWorkerDone(key);
					log.info("Freeing " + key + " for work post batch completion");
				}

				batchActor.tell(new MoreWorkMessage(masterResults), getSelf());


			}

			else {
				isDone = true;
				log.info("Done training!");
			}


		}

		else if(message instanceof AlreadyWorking) {
			final AlreadyWorking a = (AlreadyWorking) message;
			final Job j = currentJobs.get(a.getId());
			log.info("Adding job to be redistributed");
			needsToBeRedistributed.add(j);



		}

		else if(message instanceof ClearWorker) {
			log.info("Removing worker with id " + ((ClearWorker) message).getId());
			this.workers.remove(((ClearWorker) message).getId());
			this.currentJobs.remove(((ClearWorker) message).getId());
		}

		else if(message instanceof String) {
			WorkerState state = this.workers.get(message.toString());
			if(state == null) {
				state = new WorkerState(message.toString(),getSender());
				state.setAvailable(true);
				addWorker(state);
				log.info("Worker " + state.getWorkerId() + " available for work");
			}
			else {
				state.setAvailable(true);
				log.info("Worker " + state.getWorkerId() + " available for work");

			}

			getSender().tell(Ack.getInstance(),getSelf());

		}




		else if(message instanceof UpdateableImpl) {
			UpdateableImpl up = (UpdateableImpl) message;
			updates.add(up);
			log.info("Num updates so far " + updates.size() + " and partition size is " + partition);
			if(updates.size() >= partition || everyWorkerAvailable() || currentJobs.isEmpty()) 
				nextIteration();

		}

		//receive ack from worker
		else if(message instanceof Job) {
			Job j = (Job) message;
			if(!j.isDone()) {
				currentJobs.put(j.getWorkerId(),j);
				log.info("Ack from worker " + j.getWorkerId() + " on job");
			}
			else {
				log.info("Job " + j.getWorkerId() + " finished");
				setWorkerDone(j.getWorkerId());
			}

		}


		//list of examples
		else if(message instanceof List || message instanceof Pair) {

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
		this.masterResults.get().write(ds);
	}



}
