package org.deeplearning4j.iterativereduce.actor.multilayer;

import java.io.DataOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.Ack;
import org.deeplearning4j.iterativereduce.actor.core.ClusterListener;
import org.deeplearning4j.iterativereduce.actor.core.DoneMessage;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.MoreWorkMessage;
import org.deeplearning4j.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.util.SerializationUtils;
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
                .schedule(Duration.create(1,TimeUnit.MINUTES), Duration.create(1,TimeUnit.MINUTES), new Runnable() {

                    @Override
                    public void run() {
                        try {
                            List<Job> currentJobs = stateTracker.currentJobs();
                            log.info("Status check on next iteration");
                            if(stateTracker.getCurrent() == null) {
                                try {
                                    log.info("State tracker did not have a network; reinitializing");
                                    if(network == null)
                                        stateTracker.setCurrent(new UpdateableImpl(network));
                                } catch (Exception e) {
                                    throw new RuntimeException(e);
                                }
                            }

                            List<UpdateableImpl> updates = stateTracker.updates();


                            if(updates.size() >= partition || currentJobs.isEmpty())
                                nextIteration();



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
                                    partition--;
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
    public  UpdateableImpl compute(Collection<UpdateableImpl> workerUpdates,
                                   Collection<UpdateableImpl> masterUpdates) {


        DeepLearningAccumulator acc = new DeepLearningAccumulator();
        for(UpdateableImpl m : workerUpdates)
            acc.accumulate(m.get());
        UpdateableImpl masterResults = this.getResults();
        if(masterResults == null)
            masterResults = new UpdateableImpl(acc.averaged());
        else
            masterResults.set(acc.averaged());

        try {
            stateTracker.setCurrent(masterResults);

            //alert the workers to update
            for(String workerId : stateTracker.workers())
                stateTracker.addReplicate(workerId);
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





        //Wait for backend to be up

        try {
            Thread.sleep(30000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        log.info("Broadcasting initial master network");
        BaseMultiLayerNetwork network = null;
        if(this.network == null) {
            if(conf.getMultiLayerClazz().isAssignableFrom(DBN.class)) {
                network =  new DBN.Builder().withHiddenUnits(conf.getHiddenUnit()).withVisibleUnits(conf.getVisibleUnit())
                        .numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
                        .hiddenLayerSizes(conf.getLayerSizes()).renderWeights(conf.getRenderWeightEpochs())
                        .useRegularization(conf.isUseRegularization()).withDropOut(conf.getDropOut()).withLossFunction(conf.getLossFunction())
                        .withSparsity(conf.getSparsity()).useAdaGrad(conf.isUseAdaGrad()).withOptimizationAlgorithm(conf.getOptimizationAlgorithm())
                        .withMultiLayerGradientListeners(conf.getMultiLayerGradientListeners())
                        .withGradientListeners(conf.getGradientListeners())
                        .build();



            }

            else {
                network =  new BaseMultiLayerNetwork.Builder<>()
                        .numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
                        .hiddenLayerSizes(conf.getLayerSizes()).renderWeights(conf.getRenderWeightEpochs())
                        .useRegularization(conf.isUseRegularization()).withDropOut(conf.getDropOut()).withLossFunction(conf.getLossFunction())
                        .withSparsity(conf.getSparsity()).useAdaGrad(conf.isUseAdaGrad()).withOptimizationAlgorithm(conf.getOptimizationAlgorithm())
                        .withMultiLayerGradientListeners(conf.getMultiLayerGradientListeners())
                        .withGradientListeners(conf.getGradientListeners())
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





        //after worker is instantiated broadcast the master network to the worker
        mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
                masterResults), getSelf());

    }







    /**
     * Check on the next iteration
     * @throws Exception
     */
    protected void nextIteration() throws Exception {
        List<UpdateableImpl> updates = stateTracker.updates();
        if(!updates.isEmpty()) {
            UpdateableImpl masterResults = this.compute(updates, updates);


            epochsComplete++;
            //tell the batch actor to send out another dataset
            if(!isDone())
                batchActor.tell(new MoreWorkMessage(masterResults), getSelf());
            //clear previous batch
            stateTracker.updates().clear();
            log.info("Broadcasting weights");

            for(String worker : stateTracker.workers())
                stateTracker.addReplicate(worker);
        }



    }

    /**
     * Checks if done
     * @throws Exception
     */
    protected void checkDone() throws Exception {
        UpdateableImpl masterResults = null;
        List<UpdateableImpl> updates = stateTracker.updates();

        if(!updates.isEmpty()) {
            masterResults = compute(updates, updates);

            stateTracker.setCurrent(masterResults);


            epochsComplete++;
            stateTracker.updates().clear();

        }

        else
            masterResults = getMasterResults();

        if(stateTracker.isPretrain() && stateTracker.currentJobs().isEmpty()) {
            log.info("Switching to finetune mode");
            pretrain = false;
            stateTracker.moveToFinetune();
            SerializationUtils.saveObject(masterResults.get(), new File("pretrain-model.bin"));


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
            checkDone();
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
