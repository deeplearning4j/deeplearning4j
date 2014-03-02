package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.actor.core.FinetuneMessage;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.multilayer.MasterActor;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.scaleout.iterativereduce.multi.gradient.UpdateableGradientImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.japi.Creator;


public class BatchActor extends UntypedActor {

	protected DataSetIterator iter;
	private final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	private int numTimesReset;
	private static Logger log = LoggerFactory.getLogger(BatchActor.class);
	private int maxReset = 1;
	private ScheduledExecutorService iterChecker;
	public final static String FINETUNE = "finetune";

	public BatchActor(DataSetIterator iter,int maxReset) {
		this.iter = iter;
		this.maxReset = maxReset;
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());
		mediator.tell(new DistributedPubSubMediator.Subscribe(FINETUNE, getSelf()), getSelf());
	/*	iterChecker = Executors.newScheduledThreadPool(1);
		iterChecker.scheduleAtFixedRate(new Runnable() {

			@Override
			public void run() {
				if(BatchActor.this.maxReset == numTimesReset) {
					log.info("Shutting down via batch actor and max resets");
					mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.SHUTDOWN,
							new ShutdownMessage()),mediator);
					try {
						iterChecker.awaitTermination(60,TimeUnit.SECONDS);
					} catch (InterruptedException e) {
						Thread.currentThread().interrupt();
					}
				}
			}

		}, 10,60, TimeUnit.SECONDS);*/

	}


	@Override
	public void onReceive(Object message) throws Exception {
		if(message instanceof DistributedPubSubMediator.SubscribeAck) {
			log.info("Susbcribed");
		}
		else if(message instanceof ResetMessage) {
			iter.reset();
			numTimesReset++;
		}

		else if(message instanceof FinetuneMessage) {
			FinetuneMessage m = (FinetuneMessage) message;
			UpdateableImpl result = (UpdateableImpl) m.getUpdateable();
			final UpdateableImpl save = SerializationUtils.clone(result);
			mediator.tell(new DistributedPubSubMediator.Publish(ModelSavingActor.SAVE,
					save), mediator);

			log.info("Broadcasting another dataset");
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.BROADCAST,
					result), mediator);


			if(iter.hasNext()) {
				log.info("Propagating new work to master");
				mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
						iter.next()), mediator);
			}

			
			else
				unhandled(message);
		}
	}



	public DataSetIterator getIter() {
		return iter;
	}



	public static class BatchActorFactory implements Creator<BatchActor> {

		/**
		 * 
		 */
		private static final long serialVersionUID = -2260113511909990862L;

		public BatchActorFactory(DataSetIterator iter,int maxReset) {
			if(iter == null)
				throw new IllegalArgumentException("Iter can't be null");
			this.iter = iter;
		}

		private DataSetIterator iter;
		private int maxReset = 1;
		@Override
		public BatchActor create() throws Exception {
			return new BatchActor(iter,maxReset);
		}



	}



	public int getNumTimesReset() {
		return numTimesReset;
	}





}
