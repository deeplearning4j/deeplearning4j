package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.io.DataOutputStream;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.util.ActorRefUtils;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.ComputableMaster;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.jblas.DoubleMatrix;

import scala.Option;
import scala.concurrent.Future;
import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.Cancellable;
import akka.actor.OneForOneStrategy;
import akka.actor.SupervisorStrategy;
import akka.actor.SupervisorStrategy.Directive;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterReceptionistExtension;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Function;

import com.google.common.collect.Lists;

/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public abstract class MasterActor<E extends Updateable<?>> extends UntypedActor implements DeepLearningConfigurable,ComputableMaster<E> {

    protected Conf conf;
    protected LoggingAdapter log = Logging.getLogger(getContext().system(), this);
    protected ActorRef batchActor;
    protected HazelCastStateTracker stateTracker;
    protected int epochsComplete;
    protected final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
    public static String BROADCAST = "broadcast";
    public static String MASTER = "result";
    public static String SHUTDOWN = "shutdown";
    public static String FINISH = "finish";
    Cluster cluster = Cluster.get(getContext().system());
    ClusterReceptionistExtension receptionist = ClusterReceptionistExtension.get (getContext().system());
    protected boolean isDone = false;
    protected Cancellable forceNextPhase,clearStateWorkers;

    /**
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor to use for data set distribution
     * @param stateTracker Hazel Cast State Tracker
     *
     */
    public MasterActor(Conf conf,ActorRef batchActor,HazelCastStateTracker stateTracker) {
        this.conf = conf;
        this.batchActor = batchActor;
        //subscribe to broadcasts from workers (location agnostic)



        try {
            this.stateTracker = stateTracker;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        stateTracker.runPreTrainIterations(conf.getNumPasses());


        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.MASTER, getSelf()), getSelf());
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.FINISH, getSelf()), getSelf());



    }

    /**
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor to use for data set distribution
     *
     */
    public MasterActor(Conf conf,ActorRef batchActor) {
        this(conf,batchActor,null);


    }




    @Override
    public void aroundPostRestart(Throwable reason) {
        super.aroundPostRestart(reason);
        log.info("Restarted because of ",reason);
    }

    @Override
    public void aroundPreRestart(Throwable reason, Option<Object> message) {
        super.aroundPreRestart(reason, message);
        log.info("Restarted because of ",reason + " with message " + message.toString());

    }


    @Override
    public void preStart() throws Exception {
        super.preStart();
        mediator.tell(new Put(getSelf()), getSelf());
        ActorRef self = self();
        log.info("Setup master with path " + self.path());
        log.info("Pre start on master " + this.self().path().toString());
    }




    @Override
    public void postStop() throws Exception {
        super.postStop();
        log.info("Post stop on master");
        cluster.unsubscribe(getSelf());
        if(clearStateWorkers != null)
            clearStateWorkers.cancel();
        if(forceNextPhase != null)
            forceNextPhase.cancel();
    }



    @Override
    public abstract void complete(DataOutputStream ds);

    @SuppressWarnings("unchecked")
    @Override
    public  E getResults() {
        try {
            return (E) stateTracker.getCurrent();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public SupervisorStrategy supervisorStrategy() {
        return new OneForOneStrategy(0, Duration.Zero(),
                new Function<Throwable, Directive>() {
                    public Directive apply(Throwable cause) {
                        log.error("Problem with processing",cause);
                        return SupervisorStrategy.resume();
                    }
                });
    }

    public Conf getConf() {
        return conf;
    }

    public E getMasterResults() {
        return getResults();
    }

    public boolean isDone() {
        return isDone;
    }




}
