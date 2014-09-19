package org.deeplearning4j.iterativereduce.actor.core.actor;

import akka.actor.*;
import akka.actor.SupervisorStrategy.Directive;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterReceptionistExtension;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Function;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.actor.util.ActorRefUtils;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.ComputableMaster;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.deeplearning4j.util.SerializationUtils;
import scala.Option;
import scala.concurrent.Future;
import scala.concurrent.duration.Duration;

import java.io.DataOutputStream;
import java.io.File;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;


/**
 * Handles a applyTransformToDestination of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public abstract class MasterActor<E extends Updateable<?>> extends UntypedActor implements DeepLearningConfigurable,ComputableMaster<E> {

    protected Conf conf;
    protected LoggingAdapter log = Logging.getLogger(getContext().system(), this);
    protected ActorRef batchActor;
    protected StateTracker<E> stateTracker;
    protected int epochsComplete;
    protected AtomicLong oneDown;
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
     * @param batchActor the batch actor to use for data applyTransformToDestination distribution
     * @param tracker the state tracker
     *
     */
    public MasterActor(Conf conf,ActorRef batchActor,StateTracker<E> tracker) {
        this.conf = conf;
        this.batchActor = batchActor;

        //subscribe to broadcasts from workers (location agnostic)



        try {
            this.stateTracker = tracker;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        stateTracker.runPreTrainIterations(conf.getNumPasses());


        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.MASTER, getSelf()), getSelf());
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.FINISH, getSelf()), getSelf());

/*
		 * Ensures there's no one off errors by forcing the next phase as well as ensures
		 * that if the system is done it shuts down
		 */
        forceNextPhase =  context().system().scheduler()
                .schedule(Duration.create(10, TimeUnit.SECONDS), Duration.create(10,TimeUnit.SECONDS), new Runnable() {

                    @Override
                    public void run() {
                        if(stateTracker.isDone())
                            return;

                        try {
                            List<Job> currentJobs = stateTracker.currentJobs();
                            log.info("Status check on next iteration");


                            Collection<String> updates = stateTracker.workerUpdates();
                            if(currentJobs.size() == 1 && oneDown != null) {
                                long curr = TimeUnit.MILLISECONDS.toMinutes(System.currentTimeMillis() - oneDown.get());
                                if(curr >= 5) {
                                    stateTracker.currentJobs().clear();
                                    oneDown = null;
                                    log.info("Clearing out stale jobs");
                                }
                            }

                            else if(currentJobs.size() == 1) {
                                log.info("Marking start of stale jobs");
                                oneDown = new AtomicLong(System.currentTimeMillis());
                            }

                            if(updates.size() >= stateTracker.workers().size() || currentJobs.isEmpty())
                                nextBatch();

                            else
                                log.info("Still waiting on next batch, so far we have updates of size: " + updates.size()  + " out of " + stateTracker.workers().size());

                            log.info("Current jobs left " + currentJobs);


                        }catch(Exception e) {
                            throw new RuntimeException(e);
                        }

                    }

                }, context().dispatcher());

        this.clearStateWorkers =  context().system().scheduler()
                .schedule(Duration.create(1,TimeUnit.MINUTES), Duration.create(1,TimeUnit.MINUTES), new Runnable() {

                    @Override
                    public void run() {
                        if(stateTracker.isDone())
                            return;

                        try {
                            long now = System.currentTimeMillis();
                            Map<String,Long> heartbeats = MasterActor.this.stateTracker.getHeartBeats();
                            for(String key : heartbeats.keySet()) {
                                long lastChecked = heartbeats.get(key);
                                long diff = now - lastChecked;
                                long seconds = TimeUnit.MILLISECONDS.toSeconds(diff);
                                if(seconds >= 120) {
                                    log.info("Removing stale worker " + key);
                                    MasterActor.this.stateTracker.removeWorker(key);
                                }

                            }



                        }catch(Exception e) {
                            throw new RuntimeException(e);
                        }

                    }

                }, context().dispatcher());


    }




    /**
     * Checks if done
     * @throws Exception
     */
    protected void nextBatch() throws Exception {
        Collection<String> updates = stateTracker.workerUpdates();
        //ensure there aren't any jobs still in progress
        if(!updates.isEmpty() && stateTracker.currentJobs().isEmpty()) {
            E masterResults = compute();
            log.info("Updating next batch");
            stateTracker.setCurrent(masterResults);
            for(String s : stateTracker.workers()) {
                log.info("Replicating new network to " + s);
                stateTracker.addReplicate(s);
                stateTracker.enableWorker(s);

            }
            epochsComplete++;
            stateTracker.workerUpdates().clear();
            while(masterResults == null) {
                log.info("On next batch master results was null, attempting to grab results again");
                masterResults = getResults();
            }



            //tell the batch actor to send more work

            Future<Void> f = Futures.future(new Callable<Void>() {
                /**
                 * Computes a result, or throws an exception if unable to do so.
                 *
                 * @return computed result
                 * @throws Exception if unable to compute a result
                 */
                @Override
                public Void call() throws Exception {
                    mediator.tell(new DistributedPubSubMediator.Publish(BatchActor.BATCH,
                            MoreWorkMessage.getInstance() ), getSelf());

                    log.info("Requesting more work...");
                    return null;
                }
            },context().dispatcher());

            ActorRefUtils.throwExceptionIfExists(f,context().dispatcher());

        }



    }

    protected void doDoneOrNextPhase() throws Exception {
        E masterResults = null;
        Collection<String> updates = stateTracker.workerUpdates();

        if(!updates.isEmpty()) {
            masterResults = compute();

            stateTracker.setCurrent(masterResults);


            epochsComplete++;
            stateTracker.workerUpdates().clear();

        }

        else
            masterResults = getMasterResults();


        while(!stateTracker.currentJobs().isEmpty()) {
            log.info("Waiting fo jobs to finish up before next phase...");
            Thread.sleep(30000);
        }

        if(stateTracker.isPretrain() && stateTracker.currentJobs().isEmpty()) {
            log.info("Switching to finetune mode");
            stateTracker.moveToFinetune();
            SerializationUtils.saveObject(masterResults.get(), new File("pretrain-model.bin"));


            while(masterResults == null) {
                masterResults = getMasterResults();
            }


            mediator.tell(new DistributedPubSubMediator.Publish(BatchActor.BATCH,
                    ResetMessage.getInstance() ), getSelf());
            mediator.tell(new DistributedPubSubMediator.Publish(BatchActor.BATCH,
                    MoreWorkMessage.getInstance() ), getSelf());



            batchActor.tell(ResetMessage.getInstance(), getSelf());
            batchActor.tell(MoreWorkMessage.getInstance(), getSelf());

        }

        else if(stateTracker.currentJobs().isEmpty()) {
            isDone = true;
            stateTracker.finish();
            log.info("Done training!");
        }

    }



    /**
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor to use for data applyTransformToDestination distribution
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
        return new OneForOneStrategy(0,
                Duration.Zero(),
                new Function<Throwable, Directive>() {
                    public Directive apply(Throwable cause) {
                        log.error("Problem with processing",cause);
                        return SupervisorStrategy.resume();
                    }
                }
            );
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
