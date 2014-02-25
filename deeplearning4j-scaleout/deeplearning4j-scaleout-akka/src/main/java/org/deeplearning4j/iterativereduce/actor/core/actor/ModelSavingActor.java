package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

import org.deeplearning4j.nn.Persistable;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Creator;



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

	public ModelSavingActor(String pathToSave) {
		this.pathToSave = pathToSave;
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
			File save = new File(pathToSave);
			if(save.exists()) {
				File parent = save.getParentFile();
				save.renameTo(new File(parent,save.getName() + "-" + System.currentTimeMillis()));
			}
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(save));
			u.get().write(bos);
			bos.flush();
			bos.close();
			log.info("saved model to " + pathToSave);


		}
		
		else
			unhandled(message);
	}


	public static class ModelSavingActorFactory implements Creator<ModelSavingActor> {

		private static final long serialVersionUID = 6450982780084088162L;
		private String path;


		public ModelSavingActorFactory(String path) {
			this.path = path;
		}

		@Override
		public ModelSavingActor create() throws Exception {
			return new ModelSavingActor(path);
		}

	}


}
