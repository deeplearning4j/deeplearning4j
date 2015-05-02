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

package org.deeplearning4j.iterativereduce.runtime.yarn.appmaster;


import java.io.FileInputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.ResourceRequest;
import org.apache.hadoop.yarn.exceptions.YarnRemoteException;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.deeplearning4j.iterativereduce.runtime.ComputableMaster;
import org.deeplearning4j.iterativereduce.runtime.ConfigFields;
import org.deeplearning4j.iterativereduce.runtime.Utils;
import org.deeplearning4j.iterativereduce.runtime.yarn.ContainerManagerHandler;
import org.deeplearning4j.iterativereduce.runtime.yarn.ResourceManagerHandler;
import org.deeplearning4j.iterativereduce.runtime.yarn.avro.generated.FileSplit;
import org.deeplearning4j.iterativereduce.runtime.yarn.avro.generated.StartupConfiguration;
import org.deeplearning4j.iterativereduce.runtime.yarn.avro.generated.WorkerId;
import org.deeplearning4j.scaleout.api.ir.Updateable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 * Future YARN entry point
 */
public class ApplicationMaster<T extends Updateable> extends Configured
        implements Tool {

  private static final Logger LOG = LoggerFactory.getLogger(ApplicationMaster.class);

  private String masterHost;
  private int masterPort;
  private InetSocketAddress masterAddr;
  private ComputableMaster<T> masterComputable;
  private Class<T> masterUpdateable;
  private int batchSize;
  private int iterationCount;
  private Map<CharSequence, CharSequence> appConfig;

  private Configuration conf;
  private ApplicationAttemptId appAttemptId;
  private String appName;
  private Properties props;
  private Set<ConfigurationTuple> confTuples;

  private Class<?> inputFormatClass;

  private enum ReturnCode {
    OK(0), MASTER_ERROR(-1), CONTAINER_ERROR(1);

    private int code;

    ReturnCode(int code) {
      this.code = code;
    }

    public int getCode() {
      return code;
    }
  }

  public ApplicationMaster(ComputableMaster<T> computableMaster,
                           Class<T> updatable) throws IOException {

    // TODO: make port configurable
    this(9999, computableMaster, updatable);
  }

  public ApplicationMaster(int port, ComputableMaster<T> computableMaster,
                           Class<T> updatable) throws  IOException {

    //masterHost = InetAddress.getLocalHost().getHostName();
    masterHost = InetAddress.getLocalHost().getCanonicalHostName();
    masterPort = port;
    masterAddr = new InetSocketAddress(masterHost, masterPort);
    masterComputable = computableMaster;
    masterUpdateable = updatable;

    props = new Properties();
    props.load(new FileInputStream(ConfigFields.APP_CONFIG_FILE)); // Should be
    // in ./ - as
    // the Client
    // should've
    // shipped it
    ContainerId containerId = ConverterUtils.toContainerId(System
            .getenv(ApplicationConstants.AM_CONTAINER_ID_ENV));
    appAttemptId = containerId.getApplicationAttemptId();
    appName = props.getProperty(ConfigFields.APP_NAME,
            ConfigFields.DEFAULT_APP_NAME).replace(' ', '_');

    batchSize = Integer.parseInt(props.getProperty(ConfigFields.APP_BATCH_SIZE,
            "200"));
    iterationCount = Integer.parseInt(props.getProperty(
            ConfigFields.APP_ITERATION_COUNT, "1"));

    String inputFormatClassString = props.getProperty( ConfigFields.INPUT_FORMAT_CLASS, ConfigFields.INPUT_FORMAT_CLASS_DEFAULT );

    LOG.debug( "Using Input Format: " + inputFormatClassString );
    LOG.info( "IR:AppMaster > Using Input Format: " + inputFormatClassString );

    Class<?> if_class = null;
    try {
      if_class = Class.forName( inputFormatClassString );
    } catch (ClassNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

    // need to check its a legit input format subclass

    if ( null == if_class ) {

      this.inputFormatClass = TextInputFormat.class;

    } else if (InputFormat.class.isAssignableFrom(if_class) ) {

      LOG.debug( "good input format: " + inputFormatClassString );
      this.inputFormatClass = if_class;

    } else {
      LOG.debug( "bad input format: " + inputFormatClassString + ", defaulting to TextInputFormat" );
      this.inputFormatClass = TextInputFormat.class;
      // TODO: do we die here? what do we do?
    }

    // Copy all properties into appConfig to be passed down to workers, TODO:
    // fix collection merging
    appConfig = new HashMap<>();
    for (Map.Entry<Object, Object> prop : props.entrySet()) {
      appConfig.put((String) prop.getKey(), (String) prop.getValue());
    }

    if (LOG.isDebugEnabled()) {
      LOG.debug("Configuration entries: ");
      for (Map.Entry<CharSequence, CharSequence> entry : appConfig.entrySet()) {
        LOG.debug(entry.getKey() + "=" + entry.getValue());
      }

      LOG.debug("Initialized application master" + ", masterHost=" + masterHost
              + ", masterPort=" + masterPort + ", masterAddress=" + masterAddr
              + ", masterComputable=" + masterComputable.getClass().getName()
              + ", masterUpdateable=" + masterUpdateable.getClass().getName()
              + ", appAttemptId=" + appAttemptId);
    }
  }

  class ConfigurationTuple {
    private String host;
    private String workerId;
    private StartupConfiguration config;

    public ConfigurationTuple(String host, String workerId,
                              StartupConfiguration config) {
      this.host = host;
      this.workerId = workerId;
      this.config = config;

      LOG.debug("Created configuration typle" + ", host=" + this.host
              + ", workerId=" + this.workerId + ", startupConfiguration="
              + this.config);
    }

    public String getHost() {
      return host;
    }

    public String getWorkerId() {
      return workerId;
    }

    public StartupConfiguration getConfig() {
      return config;
    }
  }

  // TODO: cache this!
  private Set<ConfigurationTuple> getConfigurationTuples() throws IOException {
    if(confTuples != null)
      return confTuples;
    Path inputPath = new Path(props.getProperty(ConfigFields.APP_INPUT_PATH));
    FileSystem fs = FileSystem.get(conf);
    FileStatus f = fs.getFileStatus( inputPath );
    //BlockLocation[] bl = fs.getFileBlockLocations(p, 0, f.getLen());
    Set<ConfigurationTuple> configTuples = new HashSet<>();
    int workerId = 0;

    JobConf job = new JobConf( new Configuration() );

    job.setInputFormat( (Class<? extends InputFormat>) this.inputFormatClass ); //TextInputFormat.class);


    FileInputFormat.setInputPaths(job, inputPath);

    InputSplit[] splits =
            job.getInputFormat().getSplits( job, job.getNumMapTasks() );



    for ( InputSplit split : splits) {



      FileSplit convertedToMetronomeSplit = new FileSplit();

      org.apache.hadoop.mapred.FileSplit hadoopFileSplit = (org.apache.hadoop.mapred.FileSplit)split;

      if (hadoopFileSplit.getLength() - hadoopFileSplit.getStart() > 0) {
        convertedToMetronomeSplit.setLength(hadoopFileSplit.getLength());
        convertedToMetronomeSplit.setOffset(hadoopFileSplit.getStart());
        convertedToMetronomeSplit.setPath(hadoopFileSplit.getPath().toString());

        StartupConfiguration config = StartupConfiguration.newBuilder()
                .setBatchSize(batchSize).setIterations(iterationCount)
                .setOther(appConfig).setSplit( convertedToMetronomeSplit ).build();

        String wid = "worker-" + workerId;
        ConfigurationTuple tuple = new ConfigurationTuple( split.getLocations()[ 0 ], wid, config );

        configTuples.add(tuple);
        workerId++;

        LOG.info( "IR_AM_worker: " + wid + " added split: " + convertedToMetronomeSplit.toString() );

      } else {
        LOG.info( "IR_AM: Culled out 0 length Split: " + convertedToMetronomeSplit.toString() );
      }



    }

    LOG.info( "Total Splits/Workers: " + configTuples.size() );

    confTuples = configTuples;
    return configTuples;
  }



  private Map<WorkerId, StartupConfiguration> getMasterStartupConfiguration(
          Set<ConfigurationTuple> configTuples) {
    Map<WorkerId, StartupConfiguration> startupConfig = new HashMap<>();

    for (ConfigurationTuple tuple : configTuples) {
      WorkerId wid = Utils.createWorkerId(tuple.getWorkerId());
      startupConfig.put(wid, tuple.getConfig());
    }

    return startupConfig;
  }

  private Map<String, Integer> getNumberContainersHostMapping(
          Set<ConfigurationTuple> configTuples) {

    Map<String, Integer> containerHostMapping = new HashMap<>();

    for (ConfigurationTuple tuple : configTuples) {
      Integer count = containerHostMapping.get(tuple.getHost());

      if (count == null)
        count = 0;

      containerHostMapping.put(tuple.getHost(), ++count);
    }

    if (LOG.isDebugEnabled()) {
      LOG.debug("Created a host->numContainers mapping, with: ");
      for (Map.Entry<String, Integer> entry : containerHostMapping.entrySet()) {
        LOG.debug("host=" + entry.getKey() + ", amount=" + entry.getValue());
      }
    }

    return containerHostMapping;
  }

  private List<ResourceRequest> getRequestedContainersList(
          Set<ConfigurationTuple> configTuples, ResourceManagerHandler rmHandler)
          throws YarnRemoteException { // TODO: fix - find a way around this

    Map<String, Integer> numberContainerHostsMapping = getNumberContainersHostMapping(configTuples);
    List<ResourceRequest> requestedContainers = new ArrayList<>();
    int memory = Integer.parseInt(props.getProperty(ConfigFields.YARN_MEMORY,
            "512"));

    // Get the cluster map so we can do some assignment stuff
    rmHandler.getClientResourceManager();

    for (Map.Entry<String, Integer> entry : numberContainerHostsMapping
            .entrySet()) {
      LOG.debug("Creating a resource request for host " + entry.getKey()
              + ", with " + entry.getValue() + " containers");


      ResourceRequest request = Utils.createResourceRequest("*",
              entry.getValue(), memory);

      requestedContainers.add(request);
    }

    return requestedContainers;
  }

  private List<Thread> launchContainers(Set<ConfigurationTuple> configTuples,
                                        List<Container> allocatedContainers) {

    List<Thread> launchThreads = new ArrayList<>();
    Iterator<Container> ic = allocatedContainers.iterator();

    while (ic.hasNext()) {
      Container container = ic.next();
      Iterator<ConfigurationTuple> ict = configTuples.iterator();

      LOG.debug("Looking to match up split for container on host "
              + container.getNodeId().getHost());

      while (ict.hasNext()) {
        ConfigurationTuple tuple = ict.next();

        LOG.debug("Looking to match container host "
                + container.getNodeId().getHost() + ", with split host "
                + tuple.getHost());

        if (tuple.getHost().equals(container.getNodeId().getHost())) {
          LOG.debug("Found matching container for split");
          LaunchContainerRunnabble runnable = new LaunchContainerRunnabble(
                  tuple.getWorkerId(), container);
          Thread launchThread = new Thread(runnable);

          launchThreads.add(launchThread);
          launchThread.start();

          ict.remove();
          ic.remove();
          break;
        }
      }
    }

    // If we have leftovers, we don't have data-local assignments
    if (allocatedContainers.size() > 0) {
      LOG.debug("Unable to find specific matches for some app splits, launching remainder");
      ic = allocatedContainers.iterator();
      Iterator<ConfigurationTuple> ict = configTuples.iterator();

      while (ic.hasNext() && ict.hasNext()) {
        Container container = ic.next();
        ConfigurationTuple tuple = ict.next();

        LOG.debug("Launching split for host " + tuple.getHost()
                + " on container host " + container.getNodeId().getHost());

        LaunchContainerRunnabble runnable = new LaunchContainerRunnabble(
                tuple.getWorkerId(), container);
        Thread launchThread = new Thread(runnable);

        launchThreads.add(launchThread);
        launchThread.start();

        ic.remove();
        ict.remove();
      }
    }

    return launchThreads;
  }

  private class LaunchContainerRunnabble implements Runnable {
    String workerId;
    Container container;
    ContainerManagerHandler cmHandler;

    public LaunchContainerRunnabble(String workerId, Container container) {
      this.workerId = workerId;
      this.container = container;
    }

    @Override
    public void run() {
      LOG.debug("Launching container for worker=" + workerId + ", container="
              + container);
      // TODO: fix to make more robust (e.g. cache)
      cmHandler = new ContainerManagerHandler(conf, container);

      // Connect
      cmHandler.getContainerManager();

      // Get the local resources
      try {
        Map<String, LocalResource> localResources = Utils
                .getLocalResourcesForApplication(conf,
                        appAttemptId.getApplicationId(), appName, props,
                        LocalResourceVisibility.APPLICATION);

        List<String> commands = Utils.getWorkerCommand(conf, props, masterHost
                + ":" + masterPort, workerId);

        // Start
        cmHandler.startContainer(commands, localResources, Utils.getEnvironment(conf, props));

        // Get status
        cmHandler.getContainerStatus();
      } catch (IOException ex) {
       // Starting container, fatal
      }
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    // Set our own configuration (ToolRunner only sets it prior to calling
    // run())
    conf = getConf();

    // Our own RM Handler
    ResourceManagerHandler rmHandler = new ResourceManagerHandler(conf,
            appAttemptId);

    // Connect
    rmHandler.getAMResourceManager();

    // Register
    try {
      rmHandler.registerApplicationMaster(masterHost, masterPort);
    } catch (YarnRemoteException ex) {
      LOG.error(
              "Error encountered while trying to register application master", ex);
      return ReturnCode.MASTER_ERROR.getCode();
    }

    // Get file splits, configuration, etc.
    Set<ConfigurationTuple> configTuples;
    try {
      configTuples = getConfigurationTuples();
    } catch (IOException ex) {
      LOG.error("Error encountered while trying to generate configurations", ex);
      return ReturnCode.MASTER_ERROR.getCode();
    }
    // Needed for our master service later
    Map<WorkerId, StartupConfiguration> startupConf = getMasterStartupConfiguration(configTuples);

    // Initial containers we want, based off of the file splits
    List<ResourceRequest> requestedContainers = getRequestedContainersList(
            configTuples, rmHandler);
    List<ContainerId> releasedContainers = new ArrayList<>();

    // Send an initial allocation request
    List<Container> allocatedContainers = new ArrayList<>();
    try {
      int needed = configTuples.size();
      int got = 0;
      int maxAttempts = Integer.parseInt(props.getProperty(ConfigFields.APP_ALLOCATION_MAX_ATTEMPTS, "10"));
      int attempts = 0;

      List<Container> acquiredContainers;

      while (got < needed && attempts < maxAttempts) {
        LOG.info("Requesting containers" + ", got=" + got + ", needed="
                + needed + ", attempts=" + attempts + ", maxAttempts="
                + maxAttempts);

        acquiredContainers = rmHandler.allocateRequest(requestedContainers,
                releasedContainers).getAllocatedContainers();

        got += acquiredContainers.size();
        attempts++;

        allocatedContainers.addAll(acquiredContainers);
        acquiredContainers.clear();

        LOG.info("Got allocation response, allocatedContainers="
                + acquiredContainers.size());

        Thread.sleep(2500);
      }
    } catch (YarnRemoteException ex) {
      LOG.error("Encountered an error while trying to allocate containers", ex);
      return ReturnCode.MASTER_ERROR.getCode();
    }

    final int numContainers = configTuples.size();

    /*
     * 
     * 
     * TODO: fix this so we try N times to get enough containers!
     * 
     * 
     * 
     * 
     */
    // Make sure we got all our containers, or else bail
    if (allocatedContainers.size() < numContainers) {
      LOG.info("Unable to get required number of containers, will not continue"
              + ", needed=" + numContainers + ", allocated="
              + allocatedContainers.size());

      requestedContainers.clear(); // We don't want new containers!

      // Add containers into released list
      for (Container c : allocatedContainers) {
        releasedContainers.add(c.getId());
      }

      // Release containers
      try {
        rmHandler.allocateRequest(requestedContainers, releasedContainers);
      } catch (YarnRemoteException ex) {
        LOG.warn(
                "Encountered an error while trying to release unwanted containers",
                ex);
      }

      // Notify our handlers that we got a problem
      rmHandler.finishApplication("Unable to allocate containers, needed "
                      + numContainers + ", but got " + allocatedContainers.size(),
              FinalApplicationStatus.FAILED);
      // bail
      return ReturnCode.MASTER_ERROR.getCode();
    }



    // Launch our worker process, as we now expect workers to actally do
    // something
    LOG.info("Starting master service");
    ApplicationMasterService<T> masterService = new ApplicationMasterService<>(
            masterAddr, startupConf, masterComputable, masterUpdateable, appConfig,
            conf);

    ExecutorService executor = Executors.newSingleThreadExecutor();
    Future<Integer> masterThread = executor.submit(masterService);

    // We got the number of containers we wanted, let's launch them
    LOG.info("Launching child containers");
    List<Thread> launchThreads = launchContainers(configTuples,
            allocatedContainers);

    // Use an empty list for heartbeat purposes
    requestedContainers.clear();

    // Some local counters. Do we really need Atomic?
    AtomicInteger numCompletedContainers = new AtomicInteger();
    AtomicInteger numFailedContainers = new AtomicInteger();

    LOG.info("Waiting for containers to complete...");
    // Go into run-loop waiting for containers to finish, also our heartbeat
    while (numCompletedContainers.get() < numContainers) {
      // Don't pound the RM
      try {
        Thread.sleep(2000);
      } catch (InterruptedException ex) {
        LOG.warn("Interrupted while waiting on completed containers", ex);
        return ReturnCode.MASTER_ERROR.getCode();
      }

      // Heartbeat, effectively
      List<ContainerStatus> completedContainers;

      try {
        completedContainers = rmHandler.allocateRequest(requestedContainers,
                releasedContainers).getCompletedContainersStatuses();
      } catch (YarnRemoteException ex) {
        LOG.warn(
                "Encountered an error while trying to heartbeat to resource manager",
                ex);

        continue; // Nothing to report, probably an error / endless loop
      }

      for (ContainerStatus cs : completedContainers) {
        int exitCode = cs.getExitStatus();
        if (exitCode != 0) {
          numCompletedContainers.incrementAndGet();
          numFailedContainers.incrementAndGet();

          masterService.fail();
          executor.shutdown();

          // Force kill our application, fail fast?
          LOG.info("At least one container failed with a non-zero exit code ("
                  + exitCode + "); killing application");
          rmHandler
                  .finishApplication(
                          "Failing, due to at least container coming back with an non-zero exit code.",
                          FinalApplicationStatus.KILLED);

          return -10;
        } else {
          numCompletedContainers.incrementAndGet();
        }
      }
    }

    // All containers have completed
    // Wait for launch threads to complete (this shouldn't really happen)
    LOG.info("Containers completed");
    for (Thread launchThread : launchThreads) {
      try {
        launchThread.join(1000);
      } catch (InterruptedException ex) {
        LOG.warn("Interrupted while waiting for Launcher threads to complete",
                ex);
      }
    }

    // Ensure that our master service has completed as well
    if (!masterThread.isDone()) {
      masterService.fail();
    }

    int masterExit = masterThread.get();
    LOG.info("Master service completed with exitCode=" + masterExit);
    executor.shutdown();

    if (masterExit == 0) {




      String impersonatedUser = System.getenv("USER");

      UserGroupInformation ugi = UserGroupInformation.createRemoteUser(impersonatedUser);
      //UserGroupInformation.createProxyUser(impersonatedUser, UserGroupInformation.getLoginUser());
      ugi.doAs(new PrivilegedExceptionAction<Void>() {
        public Void run() {

          Path out = new Path(props.getProperty(ConfigFields.APP_OUTPUT_PATH));
          FileSystem fs;
          try {
            fs = out.getFileSystem(conf);

            FSDataOutputStream fos = fs.create(out);
            LOG.info("Writing master results to " + out.toString());

            masterComputable.complete(fos);

            fos.flush();
            fos.close();


          } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
          }



          return null;

          //FileSystem fs = FileSystem.get(conf);
          //fs.mkdir( out );
        }
      });
            	
 
    	/*
    	LOG.info( "Here we would try to write to " + out.toString() );
    	LOG.info( "As current user: " + UserGroupInformation.getCurrentUser().getShortUserName() );
    	LOG.info( "As login user: " + UserGroupInformation.getLoginUser().getShortUserName() );
    	
    	LOG.info( "Env Var User: " + System.getenv("USER") );
    	*/
      //LOG.info( "Ideally we'd be user: " + this.props.getProperty(  ) );

//    	for (Map.Entry<String, String> entry : this.conf) {
      //           LOG.info("ApplicationMaster->Conf: " + entry.getKey() + " = " + entry.getValue());
      //     }

    } else {
      LOG.warn("Not writing master results, as the master came back with errors!");
    }

    // Application finished
    ReturnCode rc = (numFailedContainers.get() == 0) ? ReturnCode.OK
            : ReturnCode.CONTAINER_ERROR;

    try {
      if (numFailedContainers.get() == 0) {
        rmHandler.finishApplication("Completed successfully",
                FinalApplicationStatus.SUCCEEDED);
      } else {
        String diag = "Completed with " + numFailedContainers.get()
                + " failed containers";
        rmHandler.finishApplication(diag, FinalApplicationStatus.FAILED);
      }
    } catch (YarnRemoteException ex) {
      LOG.warn(
              "Encountered an error while trying to send final status to resource manager",
              ex);
    }

    return rc.getCode();
  }
}