package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.actor;

import java.util.List;
import java.util.concurrent.Callable;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;
import scala.concurrent.duration.Duration;

import akka.actor.ActorRef;
import akka.actor.OneForOneStrategy;
import akka.actor.SupervisorStrategy;
import akka.actor.UntypedActor;
import akka.actor.SupervisorStrategy.Directive;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.japi.Creator;
import akka.japi.Function;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.ShutdownMessage;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.UpdateMessage;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.iterativereduce.ComputableWorker;
import com.ccc.deeplearning.scaleout.iterativereduce.Updateable;

public abstract class WorkerActor<E extends Updateable<?>> extends UntypedActor implements DeepLearningConfigurable,ComputableWorker<E> {
	protected DoubleMatrix combinedInput,outcomes;

	protected ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	protected E e;
	protected E results;
	private static Logger log = LoggerFactory.getLogger(WorkerActor.class);
	protected int fineTuneEpochs;
	protected int preTrainEpochs;
	protected int[] hiddenLayerSizes;
	protected int numHidden;
	protected int numVisible;
	protected int numHiddenNeurons;
	protected long seed;
	protected double learningRate;
	protected double corruptionLevel;
	protected Object[] extraParams;


	public final static String SYSTEM_NAME = "Workers";

	public WorkerActor(Conf conf) {
		setup(conf);
		//subscribe to broadcasts from workers (location agnostic)
		mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());

	}




	@SuppressWarnings("unchecked")
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			log.info("Subscribed to " + ack.toString());
		}
		else if(message instanceof ShutdownMessage) {
			log.info("Shutting down system for worker with address " + Cluster.get(context().system()).selfAddress().toString() );
			if(!context().system().isTerminated())
				context().system().shutdown();
		}

		else if(message instanceof List) {
			List<Pair<DoubleMatrix,DoubleMatrix>> input = (List<Pair<DoubleMatrix,DoubleMatrix>>) message;
			updateTraining(input);

		}

		else if(message instanceof UpdateMessage) {
			UpdateMessage<E> m = (UpdateMessage<E>) message;
			results = m.getUpdateable().get();
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
		Future<E> f = Futures.future(new Callable<E>() {

			@Override
			public E call() throws Exception {
			
				E work = compute();
				
				return work;
			}
			
		}, getContext().dispatcher());
		
		f.onComplete(new OnComplete<E>() {

			@Override
			public void onComplete(Throwable arg0, E work) throws Throwable {
				log.info("Updating parent actor...");
				//update parameters in master param server
				mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
						work), getSelf());				
			}
			
		}, context().dispatcher());
		
		
	}

	@Override
	public E compute(List<E> records) {
		return compute();
	}

	@Override
	public abstract E compute();

	@Override
	public boolean incrementIteration() {
		return false;
	}

	@Override
	public void setup(Conf conf) {
		hiddenLayerSizes = conf.getLayerSizes();
		numHidden = conf.getnOut();
		numVisible = conf.getnIn();
		numHiddenNeurons = hiddenLayerSizes.length;
		seed = conf.getSeed();

		learningRate = conf.getPretrainLearningRate();
		preTrainEpochs = conf.getPretrainEpochs();
		fineTuneEpochs = conf.getFinetuneEpochs();
		corruptionLevel = conf.getCorruptionLevel();
		extraParams = conf.getDeepLearningParams();
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
	public E getResults() {
		return results;
	}

	@Override
	public void update(E t) {
		this.e = t;
	}


	public static abstract class WorkerActorFactory<E> implements Creator<WorkerActor<Updateable<E>>> {

		/**
		 * 
		 */
		private static final long serialVersionUID = 381253681712601968L;

		public WorkerActorFactory(Conf conf) {
			this.conf = conf;
		}

		private Conf conf;

		@Override
		public abstract WorkerActor<Updateable<E>> create() throws Exception;

	}


}
