package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.ClearWorker;
import org.deeplearning4j.iterativereduce.actor.core.DoneMessage;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.actor.core.NeedsModelMessage;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.jblas.DoubleMatrix;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.routing.RoundRobinPool;


/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableImpl> {

	protected BaseMultiLayerNetwork network;
	protected Map<String,Job> currentJobs = new HashMap<String,Job>();


	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 */
	public MasterActor(Conf conf,ActorRef batchActor) {
		super(conf,batchActor);

	}

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
				.withSparsity(conf.getSparsity())
				.build() : this.network;
				masterResults = new UpdateableImpl(network);

				//after worker is instantiated broadcast the master network to the worker
				mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
						masterResults), getSelf());	



	}


	@SuppressWarnings({ "unchecked" })
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			log.info("Subscribed " + ack.toString());
		}


		else if(message instanceof WorkerState) {
			this.addWorker((WorkerState) message);
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
			batchActor.tell(new MoreWorkMessage(masterResults), getSelf());


		}

		else if(message instanceof ClearWorker) {
			log.info("Removing worker with id " + ((ClearWorker) message).getId());
			this.workers.remove(((ClearWorker) message).getId());
		}

		else if(message instanceof String) {
			WorkerState state = this.workers.get(message.toString());
			if(state == null) {
				state = new WorkerState(message.toString(),getSender());
				state.setAvailable(true);
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
			if(updates.size() >= partition) {
				masterResults = this.compute(updates, updates);

				epochsComplete++;
				//tell the batch actor to send out another dataset
				batchActor.tell(new MoreWorkMessage(masterResults), getSelf());
				updates.clear();
				log.info("Broadcasting weights");
				//replicate the network
				mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
						masterResults), getSelf());

			}



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
				currentJobs.remove(j.getWorkerId());
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
			else if(message instanceof Pair) {
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
