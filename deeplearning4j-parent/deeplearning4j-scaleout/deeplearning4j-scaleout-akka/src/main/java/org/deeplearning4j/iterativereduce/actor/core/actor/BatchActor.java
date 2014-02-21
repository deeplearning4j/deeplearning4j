package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.actor.core.FinetuneMessage;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.multilayer.MasterActor;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;
import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.japi.Creator;


public class BatchActor extends UntypedActor {

	private DataSetIterator iter;
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
		mediator.tell(new DistributedPubSubMediator.Publish(DoneReaper.REAPER,
				getSelf()), mediator);
		iterChecker = Executors.newScheduledThreadPool(1);
		iterChecker.scheduleAtFixedRate(new Runnable() {

			@Override
			public void run() {
				if(BatchActor.this.maxReset == numTimesReset) {
					log.info("Shutting down via batch actor and max resets");
					/*mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.SHUTDOWN,
							new ShutdownMessage()),mediator);*/
					try {
						iterChecker.awaitTermination(60,TimeUnit.SECONDS);
					} catch (InterruptedException e) {
						Thread.currentThread().interrupt();
					}
				}
			}

		}, 10,60, TimeUnit.SECONDS);

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
			Future<UpdateableImpl> f = Futures.future(new Callable<UpdateableImpl>() {

				@Override
				public UpdateableImpl call() throws Exception {					
					mediator.tell(new DistributedPubSubMediator.Publish(ModelSavingActor.SAVE,
							save), mediator);
					return save;
				}

			}, context().dispatcher());
			
			
			f.onComplete(new OnComplete<UpdateableImpl>() {

				@Override
				public void onComplete(Throwable arg0, UpdateableImpl arg1)
						throws Throwable {
					if(arg0 != null)
						throw arg0;
				}
			
			},context().dispatcher());
			

			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.BROADCAST,
					result), mediator);

			//This needs to happen to wait for state to propagate.
			try {
				Thread.sleep(15000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}

			if(iter.hasNext())
				mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
						iter.next()), mediator);

		}

		/*else if(message instanceof ShutdownMessage) {
			log.info("Shutting down system for worker with address " + Cluster.get(context().system()).selfAddress().toString());
			if(!context().system().isTerminated())
				context().system().shutdown();
		}*/
		else if(iter.hasNext()) {
			//start the pipeline
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
					iter.next()), mediator);

		}
		else
			unhandled(message);
		//each time the batch actor is pinged; check the status via the reaper; shutdown if done
		mediator.tell(new DistributedPubSubMediator.Publish(DoneReaper.REAPER,
				iter), mediator);
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
