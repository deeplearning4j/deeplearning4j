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
    public abstract E compute(Collection<E> workerUpdates,
                              Collection<E> masterUpdates);

    @Override
    public abstract void setup(Conf conf);





    /**
     * Delegates the list of datasets to the workers.
     * Each worker receives a portion of work and
     * changes its status to unavailable, this
     * will block till a worker is available.
     *
     * This work pull pattern ensures that no duplicate work
     * is flowing (vs publishing) and also allows for
     * proper batching of resources and input splits.
     * @param datasets the datasets to train
     * @throws Exception
     */
    protected void sendToWorkers(List<DataSet> datasets) throws Exception {
        //enable workers for sending out data
        for(String worker : stateTracker.workers())
            stateTracker.enableWorker(worker);
        int split = stateTracker.inputSplit();
        final List<List<DataSet>> splitList = Lists.partition(datasets,split);


        log.info("Found partition of size " + stateTracker.partition());
        Set<String> workerDelegated = new HashSet<>();
        for(int i = 0; i < splitList.size(); i++)  {
            final List<DataSet> wrap = splitList.get(i);
            final List<DataSet> work = new ArrayList<>(wrap);
            delegateJob(work,workerDelegated);

            log.info("Sending off work for batch " + i);


        }


    }

    private void delegateJob(List<DataSet> work,Set<String> workerDelegated) throws Exception {
        //block till there's an available worker
        log.info("Possible workers " + stateTracker.workers());

        Job j2;

        boolean sent = false;

        while(!sent) {
            //always update
            for(String s : stateTracker.workers()) {
                if(stateTracker.jobFor(s) == null && !workerDelegated.contains(s)) {
                    stateTracker.addReplicate(s);
                    //wrap in a job for additional metadata
                    j2 = new Job(s,(Serializable) work);
                    //replicate the job to hazelcast
                    stateTracker.addJobToCurrent(j2);
                    workerDelegated.add(s);
                    log.info("Delegated work to worker " + s + " with size " + work.size());
                    sent = true;
                    work = null;
                    break;

                }
            }

        }



    }


    /**
     * Splits the input such that each dataset is only one row
     * @param list the list of datasets to batch
     */
    protected void splitListIntoRows(List<DataSet> list) {
        Queue<DataSet> q = new ArrayDeque<>(list);
        list.clear();
        log.info("Splitting list in to rows...");
        while(!q.isEmpty()) {
            DataSet pair = q.poll();
            List<DoubleMatrix> inputRows = pair.getFirst().rowsAsList();
            List<DoubleMatrix> labelRows = pair.getSecond().rowsAsList();
            if(inputRows.isEmpty())
                throw new IllegalArgumentException("No input rows found");
            if(inputRows.size() != labelRows.size())
                throw new IllegalArgumentException("Label rows not equal to input rows");

            for(int i = 0; i < inputRows.size(); i++) {
                list.add(new DataSet(inputRows.get(i),labelRows.get(i)));
            }
        }
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
