package org.deeplearning4j.matrix.jblas.iterativereduce.actor.single;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.UpdateMessage;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.single.UpdateableSingleImpl;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.japi.Creator;

/**
 * Single worker actor for handling sub batches
 * of a training set.
 * @author Adam Gibson
 *
 */
public class WorkerActor extends org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.actor.WorkerActor<UpdateableSingleImpl> {
	private BaseNeuralNetwork network;
	private DoubleMatrix combinedInput;

	protected UpdateableSingleImpl workerResult;
	private ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();

	private static Logger log = LoggerFactory.getLogger(WorkerActor.class);
	public final static String SYSTEM_NAME = "Workers";

	public WorkerActor(Conf conf) {
		super(conf);
		setup(conf);
		//subscribe to broadcasts from workers (location agnostic)
		mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());
	}


	public static Props propsFor(ActorRef actor,Conf conf) {
		return Props.create(new WorkerActor.WorkerActorFactory(conf));
	}

	public static Props propsFor(Conf conf) {
		return Props.create(new WorkerActor.WorkerActorFactory(conf));
	}


	@SuppressWarnings("unchecked")
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			log.info("Subscribed to " + ack.toString());
		}
		else if(message instanceof List) {
			List<Pair<DoubleMatrix,DoubleMatrix>> input = (List<Pair<DoubleMatrix,DoubleMatrix>>) message;
			updateTraining(input);

		}

		else if(message instanceof UpdateMessage) {
			UpdateMessage<UpdateableSingleImpl> m = (UpdateMessage<UpdateableSingleImpl>) message;
			workerResult = (UpdateableSingleImpl) m.getUpdateable().get();
		}
		else
			unhandled(message);
	}

	private void updateTraining(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
		DoubleMatrix newInput = new DoubleMatrix(list.size(),list.get(0).getFirst().columns);
		DoubleMatrix newOutput = new DoubleMatrix(list.size(),list.get(0).getSecond().columns);
		for(int i = 0; i < list.size(); i++) {
			newInput.putRow(i,list.get(i).getFirst());
			newOutput.putRow(i,list.get(i).getSecond());
		}
		this.combinedInput = newInput;
		this.outcomes = newOutput;
		UpdateableSingleImpl work = compute();
		log.info("Updating parent actor...");
		//update parameters in master param server
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				work), getSelf());
	}

	@Override
	public UpdateableSingleImpl compute(List<UpdateableSingleImpl> records) {
		return compute();
	}

	@Override
	public UpdateableSingleImpl compute() {
		network.trainTillConvergence(combinedInput, learningRate, extraParams);
		return new UpdateableSingleImpl(network);
	}

	@Override
	public boolean incrementIteration() {
		return false;
	}

	@Override
	public void setup(Conf conf) {
		super.setup(conf);

		RandomGenerator rng = new MersenneTwister(conf.getSeed());
		network = new BaseNeuralNetwork.Builder<>()
				.numberOfVisible(numVisible).numHidden(numHidden)
				.withRandom(rng)
				.withClazz((Class<? extends BaseNeuralNetwork>) conf.getNeuralNetworkClazz()).build();

	}



	@Override
	public UpdateableSingleImpl getResults() {
		return workerResult;
	}

	@Override
	public void update(UpdateableSingleImpl t) {
		this.workerResult = t;
	}


	public static class WorkerActorFactory implements Creator<WorkerActor> {

		/**
		 * 
		 */
		private static final long serialVersionUID = 381253681712601968L;

		public WorkerActorFactory(Conf conf) {
			this.conf = conf;
		}

		private Conf conf;

		@Override
		public WorkerActor create() throws Exception {
			return new WorkerActor(conf);
		}

	}


}
