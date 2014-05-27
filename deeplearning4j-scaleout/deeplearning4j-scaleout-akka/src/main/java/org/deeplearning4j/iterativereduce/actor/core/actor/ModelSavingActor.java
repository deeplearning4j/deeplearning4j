package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.io.File;
import java.util.List;
import java.util.concurrent.TimeUnit;

import akka.actor.Cancellable;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.core.ModelSaver;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
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
import org.deeplearning4j.scaleout.iterativereduce.deepautoencoder.UpdateableEncoderImpl;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import scala.concurrent.duration.Duration;


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
    private StateTracker<Updateable<?>> stateTracker;


    public ModelSavingActor(String pathToSave,StateTracker<Updateable<?>> stateTracker) {
        this.pathToSave = pathToSave;
        modelSaver = new DefaultModelSaver(new File(pathToSave));
        this.stateTracker = stateTracker;
    }

    public ModelSavingActor(ModelSaver saver,StateTracker<Updateable<?>> stateTracker) {
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
            if(stateTracker.getCurrent().get().getClass().isAssignableFrom(BaseMultiLayerNetwork.class)) {
                BaseMultiLayerNetwork current = (BaseMultiLayerNetwork) stateTracker.getCurrent().get();
                if(current.getLayers() == null || current.getSigmoidLayers() == null)
                    throw new IllegalStateException("Invalid model found when prompted to save..");
                current.clearInput();
                stateTracker.setCurrent(new UpdateableImpl(current));
                if(stateTracker.hasBegun())
                    modelSaver.save(current);
            }
            else if(stateTracker.getCurrent().get().getClass().isAssignableFrom(DeepAutoEncoder.class)) {
                DeepAutoEncoder current = (DeepAutoEncoder) stateTracker.getCurrent().get();
                current.getEncoder().clearInput();
                if(current.getDecoder() != null)
                    current.getDecoder().clearInput();
                stateTracker.setCurrent(new UpdateableEncoderImpl(current));
                if(stateTracker.hasBegun())
                    modelSaver.save(current);
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
