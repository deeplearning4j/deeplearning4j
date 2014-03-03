package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.api.EpochDoneListener;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.jblas.DoubleMatrix;

import akka.actor.ActorRef;
import akka.actor.Address;
import akka.actor.Props;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubMediator;


/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableImpl> {



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


		Conf c = conf.copy();

		Address masterAddress = Cluster.get(context().system()).selfAddress();

		log.info("Starting worker");
		ActorNetworkRunner.startWorker(masterAddress,c);

		log.info("Broadcasting initial master network");

		BaseMultiLayerNetwork network = new BaseMultiLayerNetwork.Builder<>()
				.numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
				.hiddenLayerSizes(conf.getLayerSizes())
				.build();
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
		else if(message instanceof EpochDoneListener) {
			listener = (EpochDoneListener<UpdateableImpl>) message;
			log.info("Set listener");
		}

		else if(message instanceof UpdateableImpl) {
			UpdateableImpl up = (UpdateableImpl) message;
			updates.add(up);
			log.info("Num updates so far " + updates.size() + " and partition size is " + partition);
			if(updates.size() >= partition) {
				masterResults = this.compute(updates, updates);
				if(listener != null)
					listener.epochComplete(masterResults);

				epochsComplete++;
				batchActor.tell(masterResults, getSelf());
				updates.clear();
				log.info("Broadcasting weights");
				mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
						masterResults), getSelf());

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
