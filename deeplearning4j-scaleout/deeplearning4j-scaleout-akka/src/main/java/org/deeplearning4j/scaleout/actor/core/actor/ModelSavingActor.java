package org.deeplearning4j.scaleout.actor.core.actor;

import java.io.File;

import org.deeplearning4j.scaleout.actor.core.ClusterListener;
import org.deeplearning4j.scaleout.actor.core.DefaultModelSaver;
import org.deeplearning4j.scaleout.actor.core.ModelSaver;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.messages.MoreWorkMessage;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;


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
    private StateTracker stateTracker;


    public ModelSavingActor(String pathToSave,StateTracker stateTracker) {
        this.pathToSave = pathToSave;
        modelSaver = new DefaultModelSaver(new File(pathToSave));
        this.stateTracker = stateTracker;
    }

    public ModelSavingActor(ModelSaver saver,StateTracker stateTracker) {
        this.modelSaver = saver;
        this.stateTracker = stateTracker;

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
        if(message instanceof MoreWorkMessage) {
            if(stateTracker.getCurrent() != null) {
                Job j = (Job) stateTracker.getCurrent();
                modelSaver.save(j.getResult());
            }



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
