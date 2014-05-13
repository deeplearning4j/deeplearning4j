package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.DataOutputStream;
import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.DoneMessage;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.DeepLearningAccumulatorIterateAndUpdate;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.util.SetUtils;
import org.jblas.DoubleMatrix;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.PoisonPill;
import akka.actor.Props;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.routing.RoundRobinPool;


/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableImpl> {

    protected BaseMultiLayerNetwork network;


    /**
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor that handles data set dispersion
     */
    public MasterActor(Conf conf,ActorRef batchActor, final HazelCastStateTracker stateTracker) {
        super(conf,batchActor,stateTracker);
        setup(conf);
		/*
		 * Ensures there's no one off errors by forcing the next phase as well as ensures
		 * that if the system is done it shuts down
		 */
        forceNextPhase =  context().system().scheduler()
                .schedule(Duration.create(10,TimeUnit.SECONDS), Duration.create(10,TimeUnit.SECONDS), new Runnable() {

                    @Override
                    public void run() {
                      if(stateTracker.isDone())
                          return;

                        try {
                            List<Job> currentJobs = stateTracker.currentJobs();
                            log.info("Status check on next iteration");


                            Collection<String> updates = stateTracker.workerUpdates();

                            Set<String> workersLeft = new HashSet<>();
                            for(Job s : currentJobs)
                                  workersLeft.add(s.getWorkerId());

                            workersLeft = SetUtils.difference(workersLeft,updates);


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
                                if(seconds >= 30) {
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
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor for the cluster, this
     * will manage dataset dispersion
     * @param network the neural network to use
     */
    public MasterActor(Conf conf,ActorRef batchActor,BaseMultiLayerNetwork network,HazelCastStateTracker stateTracker) {
        super(conf,batchActor,stateTracker);
        this.network = network;
        setup(conf);

    }



    @Override
    public  UpdateableImpl compute() {


       DeepLearningAccumulatorIterateAndUpdate update = (DeepLearningAccumulatorIterateAndUpdate) stateTracker.updates();
       if(stateTracker.workerUpdates().isEmpty())
           return null;

        try {
            update.accumulate();

        }catch(Exception e) {
            log.debug("Unable to accumulate results",e);
            return null;
        }
        UpdateableImpl masterResults = getResults();
        if(masterResults == null)
            masterResults = update.accumulated();
        else
            masterResults.set(update.accumulated().get());

        try {
            stateTracker.setCurrent(masterResults);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        return masterResults;
    }



    @Override
    public void setup(Conf conf) {
        log.info("Starting workers");
        ActorSystem system = context().system();
        RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());
        //start local workers
        Props p = pool.props(WorkerActor.propsFor(conf,stateTracker));
        p = ClusterSingletonManager.defaultProps(p, "master", PoisonPill.getInstance(), "master");

        system.actorOf(p, "worker");





        log.info("Broadcasting initial master network");
        BaseMultiLayerNetwork network = null;
        if(this.network == null) {
            if(conf.getMultiLayerClazz().isAssignableFrom(DBN.class)) {
                network =  new DBN.Builder().withHiddenUnits(conf.getHiddenUnit()).withVisibleUnits(conf.getVisibleUnit())
                        .withVisibleUnitsByLayer(conf.getVisibleUnitByLayer()).withHiddenUnitsByLayer(conf.getHiddenUnitByLayer())
                        .activateForLayer(conf.getActivationFunctionForLayer())
                        .numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
                        .hiddenLayerSizes(conf.getLayerSizes()).renderWeights(conf.getRenderWeightEpochs())
                        .useRegularization(conf.isUseRegularization()).withDropOut(conf.getDropOut()).withLossFunction(conf.getLossFunction())
                        .withSparsity(conf.getSparsity()).useAdaGrad(conf.isUseAdaGrad()).withOptimizationAlgorithm(conf.getOptimizationAlgorithm())
                        .build();



            }

            else {
                network =  new BaseMultiLayerNetwork.Builder<>()
                        .numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
                        .hiddenLayerSizes(conf.getLayerSizes()).renderWeights(conf.getRenderWeightEpochs()).activateForLayer(conf.getActivationFunctionForLayer())
                         .useRegularization(conf.isUseRegularization()).withDropOut(conf.getDropOut()).withLossFunction(conf.getLossFunction())
                        .withSparsity(conf.getSparsity()).useAdaGrad(conf.isUseAdaGrad()).withOptimizationAlgorithm(conf.getOptimizationAlgorithm())
                        .build();

            }
        }

        else
            network = this.network;





        if(conf.getColumnMeans() != null)
            network.setColumnMeans(conf.getColumnMeans());
        if(conf.getColumnStds() != null)
            network.setColumnStds(conf.getColumnStds());

        UpdateableImpl masterResults = new UpdateableImpl(network);

        /**
         * Note that at this point we are storing an unitialized network.
         *
         *
         */
        try {
            this.stateTracker.setCurrent(masterResults);
            UpdateableImpl u2 = this.stateTracker.getCurrent();
            log.info("Stored " + u2.get());
        } catch (Exception e1) {
            throw new RuntimeException(e1);
        }

        stateTracker.setMiniBatchSize(conf.getSplit());

    }






    /**
     * Checks if done
     * @throws Exception
     */
    protected void nextBatch() throws Exception {
        Collection<String> updates = stateTracker.workerUpdates();
        //ensure there aren't any jobs still in progress
        if(!updates.isEmpty() && stateTracker.currentJobs().isEmpty()) {
            UpdateableImpl masterResults = compute();
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
            batchActor.tell(new MoreWorkMessage(masterResults),getSelf());
        }



    }

    private void doDoneOrNextPhase() throws Exception {
        UpdateableImpl masterResults = null;
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

            batchActor.tell(ResetMessage.getInstance(), getSelf());
            batchActor.tell(new MoreWorkMessage(masterResults), getSelf());

        }

        else if(stateTracker.currentJobs().isEmpty()) {
            isDone = true;
            stateTracker.finish();
            log.info("Done training!");
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


        //list of examples
        else if(message instanceof List || message instanceof DataSet) {

            if(message instanceof List) {
                List<DataSet> list = (List<DataSet>) message;
                //each pair in the matrix pairs maybe multiple rows
                splitListIntoRows(list);
                //delegate split to workers
                sendToWorkers(list);

            }

            //ensure split then send to workers
            else if(message instanceof DataSet) {
                DataSet pair = (DataSet) message;

                //split pair up in to rows to ensure parallelism
                List<DoubleMatrix> inputs = pair.getFirst().rowsAsList();
                List<DoubleMatrix> labels = pair.getSecond().rowsAsList();

                List<DataSet> pairs = new ArrayList<>();
                for(int i = 0; i < inputs.size(); i++) {
                    pairs.add(new DataSet(inputs.get(i),labels.get(i)));
                }


                sendToWorkers(pairs);

            }
        }

        else
            unhandled(message);
    }


    @Override
    public void complete(DataOutputStream ds) {
        this.getMasterResults().get().write(ds);
    }



}
