package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.io.File;

import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.core.ModelSaver;
import org.deeplearning4j.nn.Persistable;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterReceptionistExtension;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.event.Logging;
import akka.event.LoggingAdapter;



/**
 * Listens for a neural network to save
 * @author Adam Gibson
 *
 */
public class ModelSavingActor extends UntypedActor {

	public final static String SAVE = "save";
	private String pathToSave;
	private ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	private LoggingAdapter log = Logging.getLogger(getContext().system(), this);
	private Cluster cluster = Cluster.get(context().system());
	private ModelSaver modelSaver = new DefaultModelSaver();
	ClusterReceptionistExtension receptionist = ClusterReceptionistExtension.get (getContext().system());

	public ModelSavingActor(String pathToSave) {
		this.pathToSave = pathToSave;
		modelSaver = new DefaultModelSaver(new File(pathToSave));
	}

	public ModelSavingActor(ModelSaver saver) {
		this.modelSaver = saver;
	}



	{
		mediator.tell(new DistributedPubSubMediator.Subscribe(SAVE, getSelf()), getSelf());
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());

	}


	@Override
	public void postStop() throws Exception {
		super.postStop();
		log.info("Post stop on model saver");
		cluster.unsubscribe(getSelf());
	}

	@Override
	public void preStart() throws Exception {
		super.preStart();
		log.info("Pre start on model saver");
	}

	@Override
	@SuppressWarnings("unchecked")
	public void onReceive(final Object message) throws Exception {
		if(message instanceof Updateable) {
			Updateable<? extends Persistable> u = (Updateable<? extends Persistable>) message;
			modelSaver.save(u.get());
			log.info("saved model to " + pathToSave);


		}
		else if(message instanceof DistributedPubSubMediator.UnsubscribeAck || message instanceof DistributedPubSubMediator.SubscribeAck) {
			//reply
			mediator.tell(new DistributedPubSubMediator.Publish(ClusterListener.TOPICS,
					message), getSelf());	
			log.info("Sending sub/unsub over");
		}

		else
			unhandled(message);
	}





}
