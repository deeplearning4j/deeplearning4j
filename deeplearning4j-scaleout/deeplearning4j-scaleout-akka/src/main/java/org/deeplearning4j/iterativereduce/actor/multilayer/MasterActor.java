package org.deeplearning4j.iterativereduce.actor.multilayer;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.PoisonPill;
import akka.actor.Props;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.routing.RoundRobinPool;
import org.deeplearning4j.iterativereduce.actor.core.*;
import org.deeplearning4j.iterativereduce.actor.core.actor.BatchActor;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.DeepLearningAccumulatorIterateAndUpdate;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataOutputStream;
import java.util.Collection;


/**
 * Handles a applyTransformToDestination of workers and acts as a
 * parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends org.deeplearning4j.iterativereduce.actor.core.actor.MasterActor<UpdateableImpl> {
    //start with this network as a baseline
    protected BaseMultiLayerNetwork network;

    /**
     * Creates the master and the workers with this given conf
     * @param conf the neural net config to use
     * @param batchActor the batch actor that handles data applyTransformToDestination dispersion
     */
    public MasterActor(Conf conf,ActorRef batchActor, final HazelCastStateTracker stateTracker) {
        super(conf,batchActor,stateTracker);
        setup(conf);

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
        BaseMultiLayerNetwork network;
        if(this.network == null)
            network = conf.init();



        else
            network = this.network;





        network.initializeLayers(Nd4j.zeros(1, conf.getConf().getnIn()));

        UpdateableImpl masterResults = new UpdateableImpl(network);

        /**
         * Note that at this point we are
         * storing an uninitialized network.
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
                DataSet data = stateTracker.loadForWorker(worker);
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


                Job j2 = new Job(worker,data.copy());
                //replicate the job to state tracker
                stateTracker.addJobToCurrent(j2);
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
        this.getMasterResults().get().write(ds);
    }



}
