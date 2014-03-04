package org.deeplearning4j.iterativereduce.actor;

import java.io.Serializable;

import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.core.ShutdownMessage;
import org.deeplearning4j.iterativereduce.actor.multilayer.MasterActor;
import org.deeplearning4j.word2vec.iterator.Word2VecDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.japi.Creator;


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



	
	public int getNumTimesReset() {
		return numTimesReset;
	}



}
