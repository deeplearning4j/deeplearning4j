package org.deeplearning4j.iterativereduce.actor.deepautoencoder;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import org.deeplearning4j.models.featuredetectors.autoencoder.SemanticHashing;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.ClearWorker;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.deepautoencoder.UpdateableEncoderImpl;

import java.util.List;

/**
 * Iterative reduce actor for handling batch sizes
 * @author Adam Gibson
 *
 */
public class WorkerActor extends org.deeplearning4j.iterativereduce.actor.core.actor.WorkerActor<UpdateableEncoderImpl> {

    public WorkerActor(Conf conf,StateTracker<UpdateableEncoderImpl> tracker) throws Exception {
        super(conf,tracker);
        setup(conf);
        //subscribe to broadcasts from workers (location agnostic)
        mediator.tell(new Put(getSelf()), getSelf());

        //subscribe to broadcasts from master (location agnostic)
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());


        //subscribe to broadcasts from master (location agnostic)
        mediator.tell(new DistributedPubSubMediator.Subscribe(id, getSelf()), getSelf());

        heartbeat();

        tracker.addWorker(id);

    }

    public WorkerActor(ActorRef clusterClient,Conf conf,StateTracker<UpdateableEncoderImpl> tracker) throws Exception {
        super(conf,clusterClient,tracker);
        setup(conf);
        //subscribe to broadcasts from workers (location agnostic)
        mediator.tell(new Put(getSelf()), getSelf());

        //subscribe to broadcasts from master (location agnostic)
        mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());


        tracker.addWorker(id);
        //subscribe to broadcasts from master (location agnostic)
        mediator.tell(new DistributedPubSubMediator.Subscribe(id, getSelf()), getSelf());

        heartbeat();


    }



    public static Props propsFor(ActorRef actor,Conf conf,StateTracker<UpdateableEncoderImpl> tracker) {
        return Props.create(WorkerActor.class,actor,conf,tracker);
    }

    public static Props propsFor(Conf conf,StateTracker<UpdateableEncoderImpl> stateTracker) {
        return Props.create(WorkerActor.class,conf,stateTracker);
    }





    @SuppressWarnings("unchecked")
    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof DistributedPubSubMediator.SubscribeAck || message instanceof DistributedPubSubMediator.UnsubscribeAck) {
            DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
            //reply
            mediator.tell(new DistributedPubSubMediator.Publish(ClusterListener.TOPICS,
                    message), getSelf());

            log.info("Subscribed to " + ack.toString());
        }



        else if(message instanceof SemanticHashing) {
            if(results == null)
                results = new UpdateableEncoderImpl((SemanticHashing) message);
            else
                results.set((SemanticHashing) message);
            log.info("Set network");
        }

        else if(message instanceof Ack) {
            log.info("Ack from master on worker " + id);
        }


        else
            unhandled(message);
    }







    @Override
    public  UpdateableEncoderImpl compute(List<UpdateableEncoderImpl> records) {
        return compute();
    }

    @SuppressWarnings("unchecked")
    @Override
    public  UpdateableEncoderImpl compute() {

        if(tracker.isDone())
            return null;

        if(!tracker.workerEnabled(id)) {
            log.info("Worker " + id + " should be re enabled if not doing work");
            return null;
        }

        log.info("Training network on worker " + id);

        SemanticHashing network = getResults().get();
        isWorking.set(true);
        while(network == null) {
            try {
                //note that this always returns a copy
                network = tracker.getCurrent().get();
                results.set(network);
                log.info("Network is currently null");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }


        DataSet d = null;

        if(currentJob != null && tracker.workerEnabled(id)) {
            log.info("Found job for worker " + id);
            if(currentJob.getWork() instanceof List) {
                List<DataSet> l = (List<DataSet>) currentJob.getWork();
                d = DataSet.merge(l);
            }

            else
                d = (DataSet) currentJob.getWork();
        }


        else
            log.warn("No job found for " + id + " despite compute being called");



        if(currentJob  == null)
            return null;

        if(d == null) {
            throw new IllegalStateException("No job found for worker " + id);
        }

        if(conf.isNormalizeZeroMeanAndUnitVariance())
            d.normalizeZeroMeanZeroUnitVariance();
        if(conf.isScale())
            d.scale();
        if(d.getFeatureMatrix() == null || d.getLabels() == null)
            throw new IllegalStateException("Input cant be null");

        if(tracker.isPretrain()) {
            int numTries = 0;
            boolean done = false;
            while(!done && numTries < 3) {
                try {
                    log.info("Worker " + id + " pretraining");
                    network.getEncoder().pretrain(d.getFeatureMatrix(),conf.getDeepLearningParams());
                    done = true;
                }catch(Exception e) {
                    //diagnose what happened
                    if(d.getFeatureMatrix() == null) {
                        d = (DataSet) currentJob.getWork();
                    }
                    numTries++;
                }
            }

            if(!done) {
                log.warn("Worker " + id  + " failed! returning null");
                try {
                    if(!tracker.isDone())
                        tracker.clearJob(id);

                }catch(Exception e) {
                    throw new RuntimeException(e);
                }
                if(!tracker.isDone())
                    isWorking.set(false);
            }


        }

        else {
            int numTries = 0;
            boolean done = false;
            while(!done && numTries < 3) {
                try {
                    network.finetune(d.getFeatureMatrix(),conf.getFinetuneLearningRate(),conf.getFinetuneEpochs());
                    log.info("Worker " + id + " finetune");
                    done = true;
                }catch(Exception e) {
                    //diagnose what happened
                    numTries++;
                }
            }

            if(!done) {
                log.warn("Worker " + id  + " failed! returning null");
                try {
                    if(!tracker.isDone())
                        tracker.clearJob(id);

                }catch(Exception e) {
                    throw new RuntimeException(e);
                }
                if(!tracker.isDone())
                    isWorking.set(false);
            }
        }

        //job is delegated, clear so as not to cause redundancy
        try {
            if(!tracker.isDone())
                tracker.clearJob(id);

        }catch(Exception e) {
            throw new RuntimeException(e);
        }
        if(!tracker.isDone())
            isWorking.set(false);
        return new UpdateableEncoderImpl(network);
    }

    @Override
    public boolean incrementIteration() {
        return false;
    }

    @Override
    public void setup(Conf conf) {
        super.setup(conf);
    }



    @Override
    public void aroundPostStop() {
        super.aroundPostStop();
        //replicate the network
        mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
                new ClearWorker(id)), getSelf());
        heartbeat.cancel();
    }



    @Override
    public  UpdateableEncoderImpl getResults() {
        try {
            if(results == null)
                results = tracker.getCurrent();
        }catch(Exception e) {
            throw new RuntimeException(e);
        }

        return results;
    }

    @Override
    public  void update(UpdateableEncoderImpl t) {
        this.results = t;
    }







}
