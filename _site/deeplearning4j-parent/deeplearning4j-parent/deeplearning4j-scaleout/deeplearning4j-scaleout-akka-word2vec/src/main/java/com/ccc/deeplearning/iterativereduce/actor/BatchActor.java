package com.ccc.deeplearning.iterativereduce.actor;

import java.io.Serializable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.japi.Creator;

import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.ResetMessage;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.ShutdownMessage;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.multilayer.MasterActor;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIterator;

public class BatchActor extends UntypedActor implements Serializable {

	
	private static final long serialVersionUID = -2480161212284081409L;
	private Word2VecDataSetIterator iter;
	private final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	private int numTimesReset;
	private static transient Logger log = LoggerFactory.getLogger(BatchActor.class);

	public BatchActor(Word2VecDataSetIterator iter) {
		this.iter = iter;
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());

	}


	@Override
	public void onReceive(Object message) throws Exception {
		if(message instanceof ResetMessage) {
			iter.reset();
			numTimesReset++;
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

	}



	public Word2VecDataSetIterator getIter() {
		return iter;
	}



	public static class BatchActorFactory implements Creator<BatchActor> {

		/**
		 * 
		 */
		private static final long serialVersionUID = -2260113511909990862L;

		public BatchActorFactory(Word2VecDataSetIterator iter) {
			if(iter == null)
				throw new IllegalArgumentException("Iter can't be null");
			this.iter = iter;
		}

		private Word2VecDataSetIterator iter;

		@Override
		public BatchActor create() throws Exception {
			return new BatchActor(iter);
		}



	}



	public int getNumTimesReset() {
		return numTimesReset;
	}



}
