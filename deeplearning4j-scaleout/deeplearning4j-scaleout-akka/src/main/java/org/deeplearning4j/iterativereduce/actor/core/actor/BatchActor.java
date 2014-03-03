package org.deeplearning4j.iterativereduce.actor.core.actor;

import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.actor.core.FinetuneMessage;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.multilayer.MasterActor;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;

/**
 * Handles the data set iterator.
 * This includes disseminating new data sets to the cluster.
 * @author Adam Gibson
 *
 */
public class BatchActor extends UntypedActor {

	protected DataSetIterator iter;
	private final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	private static Logger log = LoggerFactory.getLogger(BatchActor.class);
	public final static String FINETUNE = "finetune";

	public BatchActor(DataSetIterator iter) {
		this.iter = iter;
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());

	}


	@Override
	public void onReceive(Object message) throws Exception {
		if(message instanceof DistributedPubSubMediator.SubscribeAck) {
			log.info("Susbcribed");
		}
		else if(message instanceof ResetMessage) {
			iter.reset();
		}

		else if(message instanceof FinetuneMessage) {
			FinetuneMessage m = (FinetuneMessage) message;
			UpdateableImpl result = (UpdateableImpl) m.getUpdateable();
			UpdateableImpl save = SerializationUtils.clone(result);
			log.info("Saving model");
			mediator.tell(new DistributedPubSubMediator.Publish(ModelSavingActor.SAVE,
					save), mediator);

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

}
