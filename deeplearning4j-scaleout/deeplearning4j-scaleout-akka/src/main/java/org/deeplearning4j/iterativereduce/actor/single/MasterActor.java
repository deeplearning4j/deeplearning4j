package org.deeplearning4j.iterativereduce.actor.single;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.core.actor.ModelSavingActor;
import org.deeplearning4j.iterativereduce.actor.core.api.EpochDoneListener;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.rng.SynchronizedRandomGenerator;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.deeplearning4j.scaleout.iterativereduce.single.UpdateableSingleImpl;
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
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableSingleImpl> {


	protected NeuralNetwork intialNetwork;
	

	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 * @param batchActor the batch actor to use for data distribution
	 * @param initialNetwork the initial neural network to use
	 */
	public MasterActor(Conf conf,ActorRef batchActor, NeuralNetwork intialNetwork) {
		super(conf,batchActor);
	}

	public static Props propsFor(Conf conf,ActorRef batchActor, NeuralNetwork intialNetwork) {
		return Props.create(MasterActor.class,conf,batchActor,intialNetwork);
	}

	

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
	public UpdateableSingleImpl compute(Collection<UpdateableSingleImpl> workerUpdates,
			Collection<UpdateableSingleImpl> masterUpdates) {


		SingleNetworkAccumulator acc = new SingleNetworkAccumulator();
		for(UpdateableSingleImpl m : workerUpdates) 
			acc.accumulate(m.get());

		masterResults.set(acc.averaged());

		return masterResults;
	}



	@Override
	public void setup(Conf conf) {
		//use the rng with the given seed
		RandomGenerator rng =  new SynchronizedRandomGenerator(new MersenneTwister(conf.getSeed()));
		@SuppressWarnings("unchecked")
		BaseNeuralNetwork network = new BaseNeuralNetwork.Builder<>()
				.withClazz((Class<? extends BaseNeuralNetwork>) conf.getNeuralNetworkClazz())
				.withRandom(rng).withL2(conf.getL2())
				.withMomentum(conf.getMomentum())
				.numberOfVisible(conf.getnIn())
				.numHidden(conf.getnOut())
				.build();
		

		context().system().actorOf(Props.create(ModelSavingActor.class,"nn-model.bin"),",model-saver");

		Address masterAddress = Cluster.get(context().system()).selfAddress();

		ActorNetworkRunner.startWorker(masterAddress,conf);
		
		
		
		
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				conf.getPretrainEpochs()), mediator);
		log.info("Setup master with epochs " + conf.getPretrainEpochs());
		masterResults = new UpdateableSingleImpl(network);
		
		log.info("Broadcasting initial master network");
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
			listener = (EpochDoneListener<UpdateableSingleImpl>) message;
			log.info("Set listener");
		}

		else if(message instanceof UpdateableSingleImpl) {
			UpdateableSingleImpl up = (UpdateableSingleImpl) message;
			updates.add(up);
			if(updates.size() == partition) {
				masterResults = this.compute(updates, updates);
				if(listener != null)
					listener.epochComplete(masterResults);
				//reset the dataset

				if(epochsComplete == conf.getPretrainEpochs()) {
					isDone = true;
					batchActor.tell(up, getSelf());
					updates.clear();
					Cluster.get(this.getContext().system()).down(Cluster.get(getContext().system()).selfAddress());
					context().system().shutdown();
					log.info("Last iteration; left cluster");
				}
				else {
					batchActor.tell(new ResetMessage(), getSelf());
					epochsComplete++;
					batchActor.tell(up, getSelf());
					updates.clear();
				}
				


			}

		}

		//broadcast new weights to workers
		else if(message instanceof Updateable) {
			mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
					message), getSelf());
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
		masterResults.get().write(ds);
	}


}
