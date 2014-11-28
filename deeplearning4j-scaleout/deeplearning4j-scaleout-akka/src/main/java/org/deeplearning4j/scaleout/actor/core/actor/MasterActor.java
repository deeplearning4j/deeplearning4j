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
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.api.ComputableMaster;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.messages.DoneMessage;
import org.deeplearning4j.scaleout.messages.MoreWorkMessage;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.DeepLearningAccumulatorIterateAndUpdate;
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
 * Handles a applyTransformToDestination of workers and acts as a
 * parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends  UntypedActor implements ComputableMaster {


    protected Configuration conf;
    protected LoggingAdapter log = Logging.getLogger(getContext().system(), this);
    protected ActorRef batchActor;
    protected StateTracker stateTracker;
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
     * @param batchActor the batch actor that handles data applyTransformToDestination dispersion
     */
    public MasterActor(Configuration conf,ActorRef batchActor, final StateTracker stateTracker) {
        this.conf = conf;
        this.batchActor = batchActor;

        //subscribe to broadcasts from workers (location agnostic)



        try {
            this.stateTracker = stateTracker;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        stateTracker.runPreTrainIterations(conf.getInt(NUM_PASSES,1));


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



    @Override
    public  Job compute() {


        DeepLearningAccumulatorIterateAndUpdate update = (DeepLearningAccumulatorIterateAndUpdate) stateTracker.updates();
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

        //list of examples
        else if(message instanceof Collection) {
            Collection<String> list = (Collection<String>) message;
            //workers to send job to
            for(String worker : list) {
                Job data = stateTracker.loadForWorker(worker);
                int numRetries = 0;
                while(data == null && numRetries < 3) {
                    data = stateTracker.loadForWorker(worker);
                    numRetries++;
                    if(data == null) {
                        Thread.sleep(10000);
                        log.info("Data still not found....sleeping for 10 seconds and trying again");
                    }
                }


                if(data == null && numRetries >= 3) {
                    log.info("No data found for worker..." + worker + " returning");
                    return;
                }


                //replicate the job to state tracker
                stateTracker.addJobToCurrent(data);
                //clear data immediately afterwards
                data = null;
                log.info("Job delegated for " + worker);
            }



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

            ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());

        }



    }

    protected void doDoneOrNextPhase() throws Exception {
        Job masterResults = null;
        Collection<String> updates = stateTracker.workerUpdates();

        if(!updates.isEmpty()) {
            masterResults = compute();

            stateTracker.setCurrent(masterResults);


            epochsComplete++;
            stateTracker.workerUpdates().clear();

        }

        else
            masterResults = getResults();


        while(!stateTracker.currentJobs().isEmpty()) {
            log.info("Waiting fo jobs to finish up before next phase...");
            Thread.sleep(30000);
        }


        if(stateTracker.currentJobs().isEmpty()) {
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
    public MasterActor(Configuration conf,ActorRef batchActor) {
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
