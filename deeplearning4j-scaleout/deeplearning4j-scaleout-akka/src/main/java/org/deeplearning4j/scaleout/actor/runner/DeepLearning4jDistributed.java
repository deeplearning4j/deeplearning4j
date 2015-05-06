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

package org.deeplearning4j.scaleout.actor.runner;

import akka.actor.*;
import akka.cluster.Cluster;
import akka.contrib.pattern.ClusterClient;
import akka.contrib.pattern.ClusterSingletonManager;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.routing.RoundRobinPool;
import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.actor.core.ClusterListener;
import org.deeplearning4j.scaleout.actor.core.ModelSaver;
import org.deeplearning4j.scaleout.actor.core.actor.BatchActor;
import org.deeplearning4j.scaleout.actor.core.actor.MasterActor;
import org.deeplearning4j.scaleout.actor.core.actor.ModelSavingActor;
import org.deeplearning4j.scaleout.actor.core.actor.WorkerActor;
import org.deeplearning4j.scaleout.actor.util.ActorRefUtils;
import org.deeplearning4j.scaleout.aggregator.INDArrayAggregator;
import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.api.workrouter.WorkRouter;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.messages.MoreWorkMessage;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.scaleout.workrouter.IterativeReduceWorkRouter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.concurrent.duration.Duration;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Controller for coordinating model training for a neural network based
 * on parameters across a cluster for akka.
 * @author Adam Gibson
 *
 */
public class DeepLearning4jDistributed implements DeepLearningConfigurable,Serializable {


    private static final long serialVersionUID = -4385335922485305364L;
    private transient ActorSystem system;
    private ActorRef mediator;
    private static final Logger log = LoggerFactory.getLogger(DeepLearning4jDistributed.class);
    private static String systemName = "ClusterSystem";
    private String type = "master";
    private Address masterAddress;
    private JobIterator iter;
    protected ActorRef masterActor;
    protected ModelSaver modelSaver;
    private transient ScheduledExecutorService exec;
    private transient StateTracker stateTracker;
    private int stateTrackerPort = -1;
    private String masterHost;
    private transient WorkRouter workRouter;



    /**
     * Master constructor
     * @param type the type (worker)
     * @param iter the dataset to use
     */
    public DeepLearning4jDistributed(String type, JobIterator iter) {
        this.type = type;
        this.iter = iter;
    }



    /**
     * Master constructor
     * @param iter the dataset to use
     */
    public DeepLearning4jDistributed(JobIterator iter,StateTracker stateTracker) {
        this("master",iter);
        this.stateTracker = stateTracker;
    }

    /**
     * Master constructor
     * @param iter the dataset to use
     */
    public DeepLearning4jDistributed(JobIterator iter) {
        this("master",iter);
    }



    /**
     * The worker constructor
     * @param type the type to use
     * @param address the address of the master
     */
    public DeepLearning4jDistributed(String type, String address) {
        this.type = type;
        URI u = URI.create(address);
        masterAddress = Address.apply(u.getScheme(), u.getUserInfo(), u.getHost(), u.getPort());
    }




    public DeepLearning4jDistributed() {
        super();
    }




    /**
     * Start a backend with the given role
     * @param joinAddress the join address
     * @param c the neural network configuration
     * @return the actor for this backend
     */
    public Address startBackend(Address joinAddress,Configuration c,JobIterator iter,StateTracker stateTracker) {

        ActorRefUtils.addShutDownForSystem(system);



        system.actorOf(Props.create(ClusterListener.class));

        try {
            Class<? extends WorkRouter> routerClazz =
                (Class<? extends WorkRouter>) Class.forName(c.get(WorkRouter.WORK_ROUTER, IterativeReduceWorkRouter.class.getName()));
            Constructor<?> constructor = routerClazz.getConstructor(StateTracker.class);
            workRouter = (WorkRouter) constructor.newInstance(stateTracker);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        workRouter.setup(c);

        ActorRef batchActor = system.actorOf(Props.create(BatchActor.class,iter,stateTracker,c,workRouter),"batch");

        log.info("Started batch actor");

        Props masterProps = Props.create(MasterActor.class,c,batchActor,stateTracker,workRouter);

		/*
		 * Starts a master: in the active state with the poison pill upon failure with the role of master
		 */
        final Address realJoinAddress = (joinAddress == null) ? Cluster.get(system).selfAddress() : joinAddress;

        c.set(MASTER_URL,realJoinAddress.toString());

        if(exec == null)
            exec = Executors.newScheduledThreadPool(2);


        Cluster cluster = Cluster.get(system);
        cluster.join(realJoinAddress);

        exec.schedule(new Runnable() {

            @Override
            public void run() {
                Cluster cluster = Cluster.get(system);
                cluster.publishCurrentClusterState();
            }

        }, 10, TimeUnit.SECONDS);

        masterActor = system.actorOf(
                ClusterSingletonManager.defaultProps(masterProps, "master", PoisonPill.getInstance(), "master"));

        log.info("Started master with address " + realJoinAddress.toString());
        c.set(MASTER_PATH,ActorRefUtils.absPath(masterActor, system));
        log.info("Set master abs path " + c.get(MASTER_PATH));

        return realJoinAddress;
    }


    @Override
    public void setup(final Configuration conf) {

        system = ActorSystem.create(systemName);
        ActorRefUtils.addShutDownForSystem(system);
        mediator = DistributedPubSubExtension.get(system).mediator();

        if(type.equals("master")) {

            if(iter == null)
                throw new IllegalStateException("Unable to initialize no dataset to iterate");

            log.info("Starting master");

            try {
                if(stateTracker == null) {
                    if(stateTrackerPort > 0)
                        stateTracker = new HazelCastStateTracker(stateTrackerPort);
                    else
                        stateTracker = new HazelCastStateTracker();
                }


                if(stateTracker.jobAggregator() == null) {
                    Class<? extends JobAggregator> clazz =
                        (Class<? extends JobAggregator>) Class.forName(conf.get(JobAggregator.AGGREGATOR, INDArrayAggregator.class.getName()));
                    JobAggregator agg = clazz.newInstance();
                    stateTracker.setJobAggregator(agg);
                }


                log.info("Started state tracker with connection string " + stateTracker.connectionString());



                masterAddress  = startBackend(null,conf,iter,stateTracker);

            } catch (Exception e1) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e1);
            }



            log.info("Starting Save saver");
            if(modelSaver == null)
                system.actorOf(Props.create(ModelSavingActor.class,"model-saver",stateTracker));
            else
                system.actorOf(Props.create(ModelSavingActor.class,modelSaver,stateTracker));



            //store it in zookeeper for service discovery
            conf.set(MASTER_URL,getMasterAddress().toString());
            conf.set(MASTER_PATH,ActorRefUtils.absPath(masterActor, system));

            //sets up the connection string for reference on the external worker
            conf.set(STATE_TRACKER_CONNECTION_STRING,stateTracker.connectionString());
            ActorRefUtils.registerConfWithZooKeeper(conf, system);


            system.scheduler().schedule(Duration.create(1, TimeUnit.MINUTES),
                    Duration.create(1, TimeUnit.MINUTES),
                    new Runnable() {

                        @Override
                        public void run() {
                            if (!system.isTerminated()) {
                                try {
                                    log.info("Current cluster members " +
                                            Cluster.get(system).readView().members());
                                } catch (Exception e) {
                                    log.warn("Tried reading cluster members during shutdown");
                                }
                            }

                        }

                    }, system.dispatcher());
        }

        else {

            log.info("Starting worker node");
            Address a = AddressFromURIString.parse(conf.get(MASTER_URL));

            Configuration c = new Configuration(conf);
            Cluster cluster = Cluster.get(system);
            cluster.join(a);

            try {
                String host = a.host().get();

                if(host == null)
                    throw new IllegalArgumentException("No host applyTransformToDestination for worker");


                String connectionString = conf.get(STATE_TRACKER_CONNECTION_STRING);
                //issue with setting the master url, fallback
                if(connectionString.contains("0.0.0.0")) {
                    if(masterHost == null)
                        throw new IllegalStateException("No master host specified and host discovery was lost due to" +
                            " improper setup on the master (related to hostname resolution) Please run the following" +
                            " command on your host: sudo hostname YOUR_HOST_NAME." +
                            " This will make your hostname resolution work correctly on master.");
                    connectionString = connectionString.replace("0.0.0.0",masterHost);
                }


                log.info("Creating state tracker with connection string "+  connectionString);
                if(stateTracker == null)
                    stateTracker = new HazelCastStateTracker(connectionString);

            } catch (Exception e1) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e1);
            }

            startWorker(c);

            system.scheduler().schedule(Duration.create(1, TimeUnit.MINUTES), Duration.create(1, TimeUnit.MINUTES), new Runnable() {

                @Override
                public void run() {
                    log.info("Current cluster members " + Cluster.get(system).readView().members());
                }

            },system.dispatcher());
            log.info("Setup worker nodes");
        }

        //only start dropwizard on the master
        if(type.equals("master")) {
            stateTracker.startRestApi();
        }

        else if(stateTracker instanceof HazelCastStateTracker)
            log.info("Not starting drop wizard; worker state detected");
    }


    public  void startWorker(Configuration conf) {

        Address contactAddress = AddressFromURIString.parse(conf.get(MASTER_URL));

        system.actorOf(Props.create(ClusterListener.class));
        log.info("Attempting to join node " + contactAddress);
        log.info("Starting workers");
        Set<ActorSelection> initialContacts = new HashSet<>();
        initialContacts.add(system.actorSelection(contactAddress + "/user/"));

        RoundRobinPool pool = new RoundRobinPool(Runtime.getRuntime().availableProcessors());

        ActorRef clusterClient = system.actorOf(ClusterClient.defaultProps(initialContacts),
                "clusterClient");


        try {
            String host = contactAddress.host().get();
            log.info("Connecting  to host " + host);
            int workers = stateTracker.numWorkers();
            if(workers <= 1)
                throw new IllegalStateException("Did not properly connect to cluster");


            log.info("Joining cluster of size " + workers);
            Class<? extends WorkerPerformerFactory> factoryClazz =
                (Class<? extends WorkerPerformerFactory>) Class.forName(conf.get(WorkerPerformerFactory.WORKER_PERFORMER));
            WorkerPerformerFactory factory = factoryClazz.newInstance();
            WorkerPerformer performer = factory.create(conf);

            Props p = pool.props(WorkerActor.propsFor(conf, stateTracker,performer));
            system.actorOf(p, "worker");

            Cluster cluster = Cluster.get(system);
            cluster.join(contactAddress);

            log.info("Worker joining cluster of " + stateTracker.workers().size());


        } catch (Exception e) {
            throw new RuntimeException(e);
        }




    }




    /**
     * Kicks off the distributed training.
     * It will grab the optimal batch size off of
     * the beginning of the dataset iterator which
     * is based on the desired mini batch size (conf.getSplit())
     *
     * and the number of initial workers in the state tracker after setup.
     *
     * For example, if you have a mini batch size of 10 and 8 workers
     *
     * the initial @link{JobIterator#next(int batches)} would be
     *
     * 80, this would be 10 per worker.
     */
    public void train() {
        log.info("Publishing to results for training");


        log.info("Started pipeline");
        //start the pipeline
        mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
                MoreWorkMessage.getInstance()), mediator);


        log.info("Published results");
        while(!stateTracker.isDone()) {
            log.info("State tracker not done...blocking");
            try {
                Thread.sleep(15000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        shutdown();
    }




    public Address getMasterAddress() {
        return masterAddress;
    }

    public StateTracker getStateTracker() {
        return stateTracker;
    }

    public  void setStateTracker(
            StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }

    /**
     *
     * Shut down this network actor
     */
    public void shutdown() {
        //order matters here, the state tracker should
        try {
            system.shutdown();

        }catch(Exception e ) {
          // do nothing
        }
        try {
            if(stateTracker != null)
                stateTracker.shutdown();
        }catch(Exception e ) {
          // do nothing
        }

    }

    public ModelSaver getModelSaver() {
        return modelSaver;
    }

    /**
     * Sets a custom model saver. This will allow custom directories
     * among other things when saving snapshots.
     * @param modelSaver the model saver to use
     */
    public  void setModelSaver(ModelSaver modelSaver) {
        this.modelSaver = modelSaver;
    }

    /**
     * Gets the state tracker port.
     * A lot of state trackers will be servers
     * that need to be bound on a port.
     * This will allow overrides per implementation of the state tracker
     * @return the state tracker port that the state tracker
     * server will bind to
     */
    public  int getStateTrackerPort() {
        return stateTrackerPort;
    }

    public  void setStateTrackerPort(int stateTrackerPort) {
        this.stateTrackerPort = stateTrackerPort;
    }

    public String getMasterHost() {
        return masterHost;
    }

    public void setMasterHost(String masterHost) {
        this.masterHost = masterHost;
    }
}
