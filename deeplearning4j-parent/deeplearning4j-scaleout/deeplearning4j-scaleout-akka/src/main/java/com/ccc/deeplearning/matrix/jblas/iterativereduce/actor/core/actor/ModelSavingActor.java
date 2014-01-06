package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.actor;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Creator;

import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.ShutdownMessage;
import com.ccc.deeplearning.nn.matrix.jblas.Persistable;
import com.ccc.deeplearning.scaleout.iterativereduce.Updateable;


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


	public ModelSavingActor(String pathToSave) {
		this.pathToSave = pathToSave;
	}


	{
		mediator.tell(new DistributedPubSubMediator.Subscribe(SAVE, getSelf()), getSelf());
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());

	}


	@Override
	public void onReceive(Object message) throws Exception {
		if(message instanceof Updateable) {
			@SuppressWarnings("unchecked")
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
		else if(message instanceof ShutdownMessage) {
			log.info("Shutting down system for worker with address " + Cluster.get(context().system()).selfAddress().toString() );
			if(!context().system().isTerminated())
				context().system().shutdown();
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
