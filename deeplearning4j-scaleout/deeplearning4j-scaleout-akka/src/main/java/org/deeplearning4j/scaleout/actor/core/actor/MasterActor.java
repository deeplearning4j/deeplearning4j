package org.deeplearning4j.scaleout.actor.core.actor;

import akka.actor.*;
import akka.actor.SupervisorStrategy.Directive;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterReceptionistExtension;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Function;
import akka.routing.RoundRobinPool;
import org.deeplearning4j.scaleout.actor.core.ClusterListener;
import org.deeplearning4j.scaleout.actor.core.protocol.Ack;
import org.deeplearning4j.scaleout.actor.util.ActorRefUtils;
import org.deeplearning4j.scaleout.api.workrouter.WorkRouter;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.api.ComputableMaster;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.messages.DoneMessage;
import org.deeplearning4j.scaleout.messages.MoreWorkMessage;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.IterateAndUpdateImpl;
import org.deeplearning4j.scaleout.workrouter.IterativeReduceWorkRouter;
import scala.Option;
import scala.concurrent.Future;
import scala.concurrent.duration.Duration;

import java.io.DataOutputStream;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;


/**
 * Handles a number of workers and acts as a
 * parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends  UntypedActor implements ComputableMaster {


    protected Configuration conf;
    protected LoggingAdapter log = Logging.getLogger(getContext().system(), this);
    protected ActorRef batchActor;
    protected StateTracker stateTracker;
    protected AtomicLong oneDown;
    protected final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
    public static String BROADCAST = "broadcast";
    public static String MASTER = "result";
    public static String SHUTDOWN = "shutdown";
    public static String FINISH = "finish";
    public  final static String NAME_SPACE = "org.deeplearning4j.scaleout.actor.core.actor";
    public final static String WAIT_FOR_WORKERS = NAME_SPACE + ".wait";
    public final static String POLL_FOR_WORK = NAME_SPACE + ".poll";
    protected int secondsPoll = 10;
    protected boolean waitForWorkers = true;
    Cluster cluster = Cluster.get(getContext().system());
    ClusterReceptionistExtension receptionist = ClusterReceptionistExtension.get (getContext().system());
    protected boolean isDone = false;
    protected Cancellable forceNextPhase,clearStateWorkers;
    protected WorkRouter workRouter;


    /**
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor that handles data applyTransformToDestination dispersion
     */
    public MasterActor(Configuration conf,ActorRef batchActor, final StateTracker stateTracker,WorkRouter router) {
        this.conf = conf;
        this.batchActor = batchActor;
        this.workRouter = router;

        //subscribe to broadcasts from workers (location agnostic)

        this.stateTracker = stateTracker;

        setup(conf);
        stateTracker.runPreTrainIterations(conf.getInt(NUM_PASSES,1));


        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.MASTER, getSelf()), getSelf());
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.FINISH, getSelf()), getSelf());

/*
		 * Ensures there's no one off errors by forcing the next phase as well as ensures
		 * that if the system is done it shuts down
		 */
        forceNextPhase =  context().system().scheduler()
                .schedule(Duration.create(secondsPoll, TimeUnit.SECONDS), Duration.create(secondsPoll,TimeUnit.SECONDS), new Runnable() {

                    @Override
                    public void run() {
                        if(stateTracker.isDone())
                            return;

                        try {


                            if(workRouter.sendWork())
                                nextBatch();


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



    @Override
    public  Job compute() {


        IterateAndUpdateImpl update = (IterateAndUpdateImpl) stateTracker.updates();
        if(stateTracker.workerUpdates().isEmpty())
            return null;

        try {
            update.accumulate();

        }catch(Exception e) {
            log.debug("Unable to accumulate results",e);
            return null;
        }

        Job masterResults = getResults();
        if(masterResults == null)
            masterResults = update.accumulated();


        try {
            stateTracker.setCurrent(masterResults);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        return masterResults;
    }



    @Override
    public void setup(Configuration conf) {
        log.info("Starting workers");
        ActorSystem system = context().system();
        RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());
        String performerFactoryClazz = conf.get(WorkerPerformerFactory.WORKER_PERFORMER);
        try {
            Class<? extends WorkerPerformerFactory> clazz = (Class<? extends WorkerPerformerFactory>) Class.forName(performerFactoryClazz);
            WorkerPerformerFactory factory = clazz.newInstance();
            WorkerPerformer performer = factory.create(conf);
            waitForWorkers = conf.getBoolean(WAIT_FOR_WORKERS,true);
            secondsPoll = conf.getInt(POLL_FOR_WORK,10);
            //start local workers
            Props p = pool.props(WorkerActor.propsFor(conf, stateTracker,performer));
            p = ClusterSingletonManager.defaultProps(p, "master", PoisonPill.getInstance(), "master");
            system.actorOf(p, "worker");

        } catch (Exception e) {
            throw new RuntimeException(e);
        }





    }





    @SuppressWarnings({ "unchecked" })
    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof DistributedPubSubMediator.SubscribeAck || message instanceof DistributedPubSubMediator.UnsubscribeAck) {
            DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
            //reply
            mediator.tell(new DistributedPubSubMediator.Publish(ClusterListener.TOPICS,
                    message), getSelf());


            log.info("Subscribed " + ack.toString());
        }



        else if(message instanceof DoneMessage) {
            log.info("Received done message");
            doDoneOrNextPhase();
        }


        else if(message instanceof String) {
            getSender().tell(Ack.getInstance(),getSelf());

        }



        else if(message instanceof MoreWorkMessage) {
            log.info("Prompted for more work, starting pipeline");
            mediator.tell(new DistributedPubSubMediator.Publish(BatchActor.BATCH,
                    MoreWorkMessage.getInstance() ), getSelf());

        }

        else
            unhandled(message);
    }



    @Override
    public void complete(DataOutputStream ds) {

    }




    /**
     * Checks if done
     * @throws Exception
     */
    protected void nextBatch() throws Exception {
        Collection<String> updates = stateTracker.workerUpdates();
        //ensure there aren't any jobs still in progress
        if(!updates.isEmpty() && stateTracker.currentJobs().isEmpty()) {
            Job masterResults = compute();
            log.info("Updating next batch");
            stateTracker.setCurrent(masterResults);
            for(String s : stateTracker.workers()) {
                log.info("Replicating new network to " + s);
                stateTracker.addReplicate(s);
                stateTracker.enableWorker(s);

            }
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

            ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());

        }



    }

    protected void doDoneOrNextPhase() throws Exception {
        Job masterResults = null;
        Collection<String> updates = stateTracker.workerUpdates();

        if(!updates.isEmpty()) {
            masterResults = compute();

            stateTracker.setCurrent(masterResults);
            stateTracker.workerUpdates().clear();

        }

        while(!stateTracker.currentJobs().isEmpty() && waitForWorkers) {
            log.info("Waiting for jobs to finish up before next phase...");
            Thread.sleep(30000);
        }


        if(stateTracker.currentJobs().isEmpty()) {
            isDone = true;
            nextBatch();
            stateTracker.finish();
            log.info("Done training!");
        }

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




    @SuppressWarnings("unchecked")
    @Override
    public  Job getResults() {
        try {
            return (Job) stateTracker.getCurrent();
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

    public Configuration getConf() {
        return conf;
    }



}
