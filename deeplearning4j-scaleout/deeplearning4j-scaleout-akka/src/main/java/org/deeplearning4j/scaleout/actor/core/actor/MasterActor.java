/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.actor.core.actor;

import akka.actor.*;
import akka.actor.SupervisorStrategy.Directive;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.dispatch.Futures;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Function;
import akka.routing.RoundRobinPool;
import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.actor.core.ClusterListener;
import org.deeplearning4j.scaleout.actor.core.protocol.Ack;
import org.deeplearning4j.scaleout.actor.util.ActorRefUtils;
import org.deeplearning4j.scaleout.api.workrouter.WorkRouter;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.messages.DoneMessage;
import org.deeplearning4j.scaleout.messages.MoreWorkMessage;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import scala.Option;
import scala.concurrent.Future;
import scala.concurrent.duration.Duration;

import java.lang.reflect.Constructor;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;


/**
 * Handles a number of workers and acts as a
 * parameter server for iterative reduce
 * @author Adam Gibson
 *kkl
 */
public class MasterActor extends  UntypedActor implements DeepLearningConfigurable {


    protected Configuration conf;
    protected LoggingAdapter log = Logging.getLogger(getContext().system(), this);
    protected ActorRef batchActor;
    protected StateTracker stateTracker;
    protected final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
    public static String BROADCAST = "broadcast";
    public static String MASTER = "result";
    public static String SHUTDOWN = "shutdown";
    public static String FINISH = "finish";
    public  final static String NAME_SPACE = "org.deeplearning4j.scaleout.actor.core.actor";
    public final static String POLL_FOR_WORK = NAME_SPACE + ".poll";
    protected int secondsPoll = 1;
    protected Cancellable forceNextPhase,clearStateWorkers;
    protected WorkRouter workRouter;
    protected AtomicBoolean doneCalled = new AtomicBoolean(false);


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
                        log.info("Heart beat on " + stateTracker.workers().size() + " workers");

                        if(stateTracker.isDone())
                            return;
                        if(workRouter.sendWork())
                            nextBatch();
                        try {
                            Set<Job> clear = new HashSet<>();
                            for(Job j : stateTracker.currentJobs()) {
                                if(stateTracker.recentlyCleared().contains(j.workerId())) {
                                    stateTracker.clearJob(j.workerId());
                                    clear.add(j);
                                    log.info("Found job that wasn't clear " + j.workerId());
                                }
                            }

                            stateTracker.currentJobs().removeAll(clear);
                            if(stateTracker.currentJobs().isEmpty())
                                stateTracker.recentlyCleared().clear();

                        } catch (Exception e) {
                            e.printStackTrace();
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
    public void setup(Configuration conf) {
        log.info("Starting workers");
        ActorSystem system = context().system();
        RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());
        String performerFactoryClazz = conf.get(WorkerPerformerFactory.WORKER_PERFORMER);
        try {
            Class<? extends WorkerPerformerFactory> clazz = (Class<? extends WorkerPerformerFactory>) Class.forName(performerFactoryClazz);

            WorkerPerformerFactory factory = null;

            try {
                Constructor<?> c = clazz.getConstructor(StateTracker.class);
                factory = (WorkerPerformerFactory) c.newInstance(stateTracker);
            }catch(NoSuchMethodException e) {
                factory = clazz.newInstance();
            }


            WorkerPerformer performer = factory.create(conf);
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



    /**
     * Checks if done
     * @throws Exception
     */
    protected void nextBatch() {
        Collection<String> updates = stateTracker.workerUpdates();
        Collection<Job> currentJobs;
        try {
            currentJobs = stateTracker.currentJobs();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }



        //ensure there aren't any jobs still in progress
        if(!updates.isEmpty() && currentJobs.isEmpty()) {

            workRouter.update();



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
                            MoreWorkMessage.getInstance()), getSelf());

                    log.info("Requesting more work...");
                    return null;
                }
            },context().dispatcher());

            ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());

        }

        else if(currentJobs.isEmpty()) {
            stateTracker.finish();
            stateTracker.shutdown();
            context().system().shutdown();
           log.info("Current jobs is empty and no more updates; terminating");
        }




    }

    protected void doDoneOrNextPhase() throws Exception {

        if(!stateTracker.workerUpdates().isEmpty())
            workRouter.update();




        if(stateTracker.currentJobs().isEmpty()) {
            if(doneCalled.get())
                return;
            else
            doneCalled.set(true);
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
        if(clearStateWorkers != null)
            clearStateWorkers.cancel();
        if(forceNextPhase != null)
            forceNextPhase.cancel();
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




}
