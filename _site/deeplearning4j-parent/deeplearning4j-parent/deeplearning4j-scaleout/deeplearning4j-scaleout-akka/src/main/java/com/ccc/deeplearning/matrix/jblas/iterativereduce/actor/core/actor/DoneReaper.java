package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.actor;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.datasets.iterator.DataSetIterator;

import akka.actor.ActorRef;
import akka.actor.PoisonPill;
import akka.actor.UntypedActor;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;

public class DoneReaper extends UntypedActor {

	private List<ActorRef> refs = new ArrayList<ActorRef>();
	private static Logger log = LoggerFactory.getLogger(DoneReaper.class);
	private ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	public final static String REAPER = "reaper";
	
	
	
	
	
	
	public DoneReaper() {
		super();
		mediator.tell(new DistributedPubSubMediator.Subscribe(REAPER, getSelf()), getSelf());

	}






	@Override
	public void onReceive(Object message) throws Exception {
	/*	if(message instanceof ActorRef) {
			ActorRef a = (ActorRef) message;
			context().watch(a);
			refs.add(a);

		}
		else if(message instanceof DataSetIterator) {
			DataSetIterator d = (DataSetIterator) message;
			if(!d.hasNext()) {
				
				for(ActorRef ref : refs) {
					ref.tell(PoisonPill.getInstance(), getSelf());
				}
			}
		}
		
		else
			unhandled(message);*/
	}

	
	
}
