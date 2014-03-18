package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.AlreadyWorking;
import org.deeplearning4j.iterativereduce.actor.core.ClearWorker;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.GiveMeMyJob;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.NoJobFound;
import org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor;
import org.deeplearning4j.iterativereduce.actor.util.ActorRefUtils;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;
import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.Cancellable;
import akka.actor.Props;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;

/**
 * Iterative reduce actor for handling batch sizes
 * @author Adam Gibson
 *
 */
public class WorkerActor extends org.deeplearning4j.iterativereduce.actor.core.actor.WorkerActor<UpdateableImpl> {
	protected BaseMultiLayerNetwork network;
	protected DoubleMatrix combinedInput;

	protected UpdateableImpl workerUpdateable;
	protected ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	protected Cancellable heartbeat;
	protected static Logger log = LoggerFactory.getLogger(WorkerActor.class);
	public final static String SYSTEM_NAME = "Workers";
	protected int numTimesReceivedNullJob = 0;


	public WorkerActor(Conf conf,StateTracker<UpdateableImpl> tracker) {
		super(conf,tracker);
		setup(conf);
		//subscribe to broadcasts from workers (location agnostic)
		mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());


		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(id, getSelf()), getSelf());

		heartbeat();



	}

	public WorkerActor(ActorRef clusterClient,Conf conf,StateTracker<UpdateableImpl> tracker) {
		super(conf,clusterClient,tracker);
		setup(conf);
		//subscribe to broadcasts from workers (location agnostic)
		mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());


		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(id, getSelf()), getSelf());

		heartbeat();

	}



	public static Props propsFor(ActorRef actor,Conf conf,HazelCastStateTracker tracker) {
		return Props.create(WorkerActor.class,actor,conf,tracker);
	}

	public static Props propsFor(Conf conf,HazelCastStateTracker stateTracker) {
		return Props.create(WorkerActor.class,conf,stateTracker);
	}

	protected void confirmWorking() {
		Job j = tracker.jobFor(id);

		//reply
		if(j != null)
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
					j), getSelf());	

		else
			log.warn("Not confirming work when none to be found");
	}



	protected void heartbeat() {
		heartbeat = context().system().scheduler().schedule(Duration.apply(10, TimeUnit.SECONDS), Duration.apply(10, TimeUnit.SECONDS), new Runnable() {

			@Override
			public void run() {
				log.info("Sending heartbeat to master");
				mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
						register()), getSelf());	

			}

		}, context().dispatcher());

	}

	@SuppressWarnings("unchecked")
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck || message instanceof DistributedPubSubMediator.UnsubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			//reply
			mediator.tell(new DistributedPubSubMediator.Publish(ClusterListener.TOPICS,
					message), getSelf());	

			log.info("Subscribed to " + ack.toString());
		}

		else if(message instanceof Job) {
			Job j = (Job) message;
			Job trackerJob = tracker.jobFor(id);
			if(trackerJob == null) {
				tracker.addJobToCurrent(j);
				log.info("Confirmation from " + j.getWorkerId() + " on work");
				List<DataSet> input = (List<DataSet>) j.getWork();
				confirmWorking();
				updateTraining(input);

			}

			else {
				//block till there's an available worker

				Job j2 = null;
				boolean redist = false;


				while(!redist) {
					List<String> ids = tracker.jobIds();

					for(String s : ids) {
						if(tracker.jobFor(s) == null) {

							//wrap in a job for additional metadata
							j2 = j;
							j2.setWorkerId(s);
							//replicate the network
							mediator.tell(new DistributedPubSubMediator.Publish(s,
									j2), getSelf());
							log.info("Delegated work to worker " + s);

							redist = true;
							break;

						}
					}
				}


			}


		}

		else if(message instanceof BaseMultiLayerNetwork) {
			setNetwork((BaseMultiLayerNetwork) message);
			log.info("Set network");
		}

		else if(message instanceof Ack) {
			log.info("Ack from master on worker " + id);
		}




		else if(message instanceof Updateable) {
			final UpdateableImpl m = (UpdateableImpl) message;
			Future<Void> f = Futures.future(new Callable<Void>() {

				@Override
				public Void call() throws Exception {

					if(m.get() == null) {
						setNetwork(tracker.getCurrent().get());

					}

					else {
						setWorkerUpdateable(m.clone());
						setNetwork(m.get());

					}

					return null;
				}

			},context().dispatcher());

			ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());


		}
		else
			unhandled(message);
	}

	protected  void updateTraining(List<DataSet> list) {
		DoubleMatrix newInput = new DoubleMatrix(list.size(),list.get(0).getFirst().columns);
		DoubleMatrix newOutput = new DoubleMatrix(list.size(),list.get(0).getSecond().columns);






		for(int i = 0; i < list.size(); i++) {
			newInput.putRow(i,list.get(i).getFirst());
			newOutput.putRow(i,list.get(i).getSecond());
		}

		setCombinedInput(newInput);
		setOutcomes(newOutput);

		Future<UpdateableImpl> f = Futures.future(new Callable<UpdateableImpl>() {

			@Override
			public UpdateableImpl call() throws Exception {


				UpdateableImpl work = compute();

				if(work != null) {
					log.info("Updating parent actor...");
					//update parameters in master param server
					mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
							work), getSelf());	

				}

				else {
					//ensure next iteration happens by decrementing number of required batches
					mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
							NoJobFound.getInstance()), getSelf());
					log.info("No job found; unlocking worker "  + id);
				}

				return work;
			}

		}, getContext().dispatcher());

		ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());
	}




	@Override
	public  UpdateableImpl compute(List<UpdateableImpl> records) {
		return compute();
	}

	@Override
	public  UpdateableImpl compute() {
		log.info("Training network");
		BaseMultiLayerNetwork network = this.getNetwork();
		while(network == null) {
			try {
				network = tracker.getCurrent().get();
				this.network = network;
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}


		DataSet d = null;
		Job j = tracker.jobFor(id);

		if(j != null) {
			log.info("Found job for worker " + id);
			if(j.getWork() instanceof List) {
				List<DataSet> l = (List<DataSet>) j.getWork();
				d = DataSet.merge(l);
			}

			else
				d = (DataSet) j.getWork();
			combinedInput = d.getFirst();
			outcomes = d.getSecond();
		}



		if(j == null)
			return null;

		if(d == null) {
			throw new IllegalStateException("No job found for worker " + id);
		}


		if(tracker.isPretrain()) {
			log.info("Worker " + id + " pretraining");
			network.pretrain(d.getFirst(), extraParams);
		}

		else {
			network.setInput(d.getFirst());
			log.info("Worker " + id + " finetuning");
			network.finetune(d.getSecond(), learningRate, fineTuneEpochs);
		}


		try {
			if(j != null)
				tracker.clearJob(j);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}



		return new UpdateableImpl(network);
	}

	@Override
	public boolean incrementIteration() {
		return false;
	}

	@Override
	public void setup(Conf conf) {
		super.setup(conf);
	}



	@Override
	public void aroundPostStop() {
		super.aroundPostStop();
		//replicate the network
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				new ClearWorker(id)), getSelf());
		heartbeat.cancel();
	}



	@Override
	public  UpdateableImpl getResults() {
		return workerUpdateable;
	}

	@Override
	public  void update(UpdateableImpl t) {
		this.workerUpdateable = t;
	}


	public  synchronized BaseMultiLayerNetwork getNetwork() {
		return network;
	}


	public  void setNetwork(BaseMultiLayerNetwork network) {
		this.network = network;
	}


	public  DoubleMatrix getCombinedInput() {
		return combinedInput;
	}


	public  void setCombinedInput(DoubleMatrix combinedInput) {
		this.combinedInput = combinedInput;
	}





	public  UpdateableImpl getWorkerUpdateable() {
		return workerUpdateable;
	}


	public  void setWorkerUpdateable(UpdateableImpl workerUpdateable) {
		this.workerUpdateable = workerUpdateable;
	}






}
