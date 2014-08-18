package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import akka.actor.*;
import akka.dispatch.Futures;
import org.deeplearning4j.iterativereduce.actor.core.ClearWorker;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.util.ActorRefUtils;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.ComputableWorker;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.concurrent.Future;
import scala.concurrent.duration.Duration;
import akka.actor.SupervisorStrategy.Directive;
import akka.cluster.Cluster;
import akka.cluster.ClusterEvent.MemberEvent;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.japi.Function;

/**
 * Baseline worker actor class
 * @author Adam Gibson
 *
 * @param <E>
 */
public abstract class WorkerActor<E extends Updateable<?>> extends UntypedActor implements DeepLearningConfigurable,ComputableWorker<E> {

    protected E results;
    protected Job currentJob;
    protected String id;
    Cluster cluster = Cluster.get(getContext().system());
    protected ActorRef clusterClient;
    protected String masterPath;
    protected StateTracker<E> tracker;
    protected AtomicBoolean isWorking = new AtomicBoolean(false);
    protected Conf conf;
    protected ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
    protected Cancellable heartbeat;
    protected static Logger log = LoggerFactory.getLogger(WorkerActor.class);



    public WorkerActor(Conf conf,StateTracker<E> tracker)throws Exception   {
        this(conf,null,tracker);
    }

    public WorkerActor(Conf conf,ActorRef client,StateTracker<E> tracker) throws Exception {
        setup(conf);

        this.tracker = tracker;

        //subscribe to broadcasts from workers (location agnostic)
        mediator.tell(new Put(getSelf()), getSelf());

        //subscribe to broadcasts from master (location agnostic)
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());
        //subscribe to shutdown messages
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());
        id = generateId();
        //replicate the network
        mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
                register()), getSelf());

        this.clusterClient = client;

        //ensure worker is available to tracker
        tracker.availableForWork(id);
        //master lookup
        masterPath = conf.getMasterAbsPath();
        log.info("Registered with master " + id + " at master " + conf.getMasterAbsPath());

        heartbeat();

        tracker.addWorker(id);


    }



    protected void heartbeat() throws Exception {
        heartbeat = context().system().scheduler().schedule(Duration.apply(30, TimeUnit.SECONDS), Duration.apply(30, TimeUnit.SECONDS), new Runnable() {

            @Override
            public void run() {
                if(!tracker.isDone())
                    tracker.addWorker(id);

                if(!tracker.isDone() && tracker.needsReplicate(id)) {
                    try {
                        log.info("Updating worker " + id);
                        E u = tracker.getCurrent();

                        if(u == null || u.get() == null) {
                            return;
                        }

                        results = u;
                        tracker.doneReplicating(id);
                    }catch(Exception e) {
                        throw new RuntimeException(e);
                    }
                }

                //eventually consistent storage
                try {
                    checkJobAvailable();


                    if(currentJob != null && !isWorking.get() && tracker.jobFor(id) != null) {
                        log.info("Confirmation from " + currentJob.getWorkerId() + " on work");
                        if(currentJob.getWork() == null)
                            throw new IllegalStateException("Work for worker " + id + " was null");

                        DataSet data = (DataSet) currentJob.getWork();
                        processDataSet(data.asList());

                    }

                    else if(currentJob == null || !isWorking.get() && tracker.jobFor(id) != null) {
                        if(tracker.jobFor(id) != null)
                            tracker.clearJob(id);
                        log.info("Clearing stale job... " + id);
                    }


                }catch(Exception e) {
                    throw new RuntimeException(e);
                }



            }

        }, context().dispatcher());

    }

    /* Run compute on the data applyTransformToDestination */
    protected  void processDataSet(final List<DataSet> list) {
        if(list == null || list.isEmpty()) {
            log.warn("Worker " + id + " was passed an empty or null list");
            return;
        }


        Future<E> f = Futures.future(new Callable<E>() {

            @Override
            public E call() throws Exception {

                INDArray newInput = NDArrays.create(list.size(), list.get(0).getFeatureMatrix().columns());
                INDArray newOutput = NDArrays.create(list.size(), list.get(0).getLabels().columns());


                for (int i = 0; i < list.size(); i++) {
                    newInput.putRow(i, list.get(i).getFeatureMatrix());
                    newOutput.putRow(i, list.get(i).getLabels());
                }

                //flag that work has begun if not flagged already
                tracker.beginTraining();

                if (tracker.needsReplicate(id)) {
                    log.info("Updating network for worker " + id);
                    results = tracker.getCurrent();
                    tracker.doneReplicating(id);
                }

                E work = compute();

                if (work != null) {
                    log.info("Done working; adding update to mini batch on worker " + id);
                    //update parameters in master param server
                    tracker.addUpdate(id, work);
                    //disable the worker till next batch
                    tracker.disableWorker(id);
                    log.info("Number of updates so far " + tracker.workerUpdates().size());
                }


                return work;
            }

        }, getContext().dispatcher());

        ActorRefUtils.throwExceptionIfExists(f, context().dispatcher());
    }



    /**
     * Returns a worker state with the id generated by this worker
     * @return a worker state with the id of this worker
     */
    public WorkerState register() {
        return new WorkerState(this.id);
    }

    /**
     * Generates an id for this worker
     * @return a UUID for this worker
     */
    public String generateId() {
        String base = UUID.randomUUID().toString();
        String host = System.getProperty("akka.remote.netty.tcp.hostname","localhost");
        return host + "-" + base;
    }


    @Override
    public void postStop() throws Exception {
        super.postStop();
        try {
            tracker.removeWorker(id);

        }catch(Exception e) {
            log.info("Tracker already shut down");
        }
        log.info("Post stop on worker actor");
        cluster.unsubscribe(getSelf());
    }

    @Override
    public void preStart() throws Exception {
        super.preStart();
        cluster.subscribe(getSelf(), MemberEvent.class);
        log.info("Pre start on worker");

    }

    protected void checkJobAvailable() throws Exception {
        Job j;

        if((j = tracker.jobFor(id)) == null || !tracker.workerEnabled(id)) {
            //inconsistent state
            if(!isWorking.get() && j != null)  {
                tracker.clearJob(id);
                log.info("Clearing stale job " + id);
            }

            return;
        }

        if(tracker.needsReplicate(id)) {
            try {
                log.info("Updating worker " + id);
                results = tracker.getCurrent();
                tracker.doneReplicating(id);
            }catch(Exception e) {
                throw new RuntimeException(e);
            }
        }


        if(j != null && currentJob == null) {
            log.info("Assigning job for worker " + id);
            currentJob = j;
            //clear data, no point in keeping both in memory
            tracker.updateJob(new Job(id,null));

        }

    }





    @Override
    public E compute(List<E> records) {
        return compute();
    }

    @Override
    public abstract E compute();

    @Override
    public boolean incrementIteration() {
        return false;
    }

    @Override
    public void setup(Conf conf) {
        this.conf = conf;
        String url = conf.getMasterUrl();
        this.masterPath = conf.getMasterAbsPath();
        Address a = AddressFromURIString.apply(url);
        Cluster.get(context().system()).join(a);

        mediator = DistributedPubSubExtension.get(getContext().system()).mediator();

    }



    @Override
    public SupervisorStrategy supervisorStrategy() {
        return new OneForOneStrategy(0, Duration.Zero(),
                new Function<Throwable, Directive>() {
                    public Directive apply(Throwable cause) {
                        log.error("Problem with processing",cause);
                        mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
                                new ClearWorker(id)), getSelf());


                        return SupervisorStrategy.restart();
                    }
                });
    }



    @Override
    public E getResults() {
        return results;
    }

    @Override
    public void update(E t) {
        this.results = t;
    }







}
