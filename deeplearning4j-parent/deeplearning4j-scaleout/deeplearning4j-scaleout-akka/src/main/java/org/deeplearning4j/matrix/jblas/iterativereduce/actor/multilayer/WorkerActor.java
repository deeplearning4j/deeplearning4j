package org.deeplearning4j.matrix.jblas.iterativereduce.actor.multilayer;

import java.util.List;
import java.util.concurrent.Callable;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.UpdateMessage;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.actor.MasterActor;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.rng.SynchronizedRandomGenerator;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;
import scala.concurrent.duration.Duration;

import akka.actor.ActorRef;
import akka.actor.OneForOneStrategy;
import akka.actor.Props;
import akka.actor.SupervisorStrategy;
import akka.actor.SupervisorStrategy.Directive;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.japi.Creator;
import akka.japi.Function;


public class WorkerActor extends org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.actor.WorkerActor<UpdateableImpl> {
	private BaseMultiLayerNetwork network;
	private DoubleMatrix combinedInput;

	protected UpdateableImpl workerMatrix;
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
			UpdateMessage<UpdateableImpl> m = (UpdateMessage<UpdateableImpl>) message;
			setWorkerMatrix(m.getUpdateable().get());
		}
		else
			unhandled(message);
	}

	private synchronized void updateTraining(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
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

				return work;
			}

		}, getContext().dispatcher());

		f.onComplete(new OnComplete<UpdateableImpl>() {

			@Override
			public void onComplete(Throwable arg0, UpdateableImpl work) throws Throwable {
				if(arg0 != null)
					throw arg0;

				log.info("Updating parent actor...");
				//update parameters in master param server
				mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
						work), getSelf());				
			}

		}, context().dispatcher());


	}

	@Override
	public UpdateableImpl compute(List<UpdateableImpl> records) {
		return compute();
	}

	@Override
	public UpdateableImpl compute() {
		log.info("Training network");
		network.trainNetwork(this.getCombinedInput(),this.getOutcomes(),extraParams);
		return new UpdateableImpl(network);
	}

	@Override
	public boolean incrementIteration() {
		return false;
	}

	@Override
	public void setup(Conf conf) {
		super.setup(conf);

		RandomGenerator rng = new SynchronizedRandomGenerator(new MersenneTwister(conf.getSeed()));
		network = new BaseMultiLayerNetwork.Builder<>().disableBackProp()
				.numberOfInputs(numVisible).numberOfOutPuts(numHidden).withActivation(conf.getFunction())
				.hiddenLayerSizes(hiddenLayerSizes).withRng(rng).transformWeightsAt(conf.getWeightTransforms())
				.withClazz(conf.getMultiLayerClazz()).build();

	}


	@Override
	public SupervisorStrategy supervisorStrategy() {
		return new OneForOneStrategy(0, Duration.Zero(),
				new Function<Throwable, Directive>() {
			public Directive apply(Throwable cause) {
				log.error("Problem with processing",cause);
				return SupervisorStrategy.stop();
			}
		});
	}


	@Override
	public synchronized UpdateableImpl getResults() {
		return workerMatrix;
	}

	@Override
	public synchronized void update(UpdateableImpl t) {
		this.workerMatrix = t;
	}


	public synchronized BaseMultiLayerNetwork getNetwork() {
		return network;
	}


	public synchronized void setNetwork(BaseMultiLayerNetwork network) {
		this.network = network;
	}


	public synchronized DoubleMatrix getCombinedInput() {
		return combinedInput;
	}


	public synchronized void setCombinedInput(DoubleMatrix combinedInput) {
		this.combinedInput = combinedInput;
	}


	public synchronized UpdateableImpl getWorkerMatrix() {
		return workerMatrix;
	}


	public synchronized void setWorkerMatrix(UpdateableImpl workerMatrix) {
		this.workerMatrix = workerMatrix;
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
