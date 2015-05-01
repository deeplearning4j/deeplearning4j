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

package org.deeplearning4j.iterativereduce.runtime.yarn;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.AMRMProtocol;
import org.apache.hadoop.yarn.api.ClientRMProtocol;
import org.apache.hadoop.yarn.api.protocolrecords.*;
import org.apache.hadoop.yarn.api.records.*;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnRemoteException;
import org.apache.hadoop.yarn.ipc.YarnRPC;
import org.apache.hadoop.yarn.util.Records;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;


public class ResourceManagerHandler {

  private static final Logger LOG = LoggerFactory.getLogger(ResourceManagerHandler.class);
  
  private Configuration conf;
  private ApplicationAttemptId appAttemptId;

  private AMRMProtocol amResourceManager;
  private ClientRMProtocol clientResourceManager;
  private AtomicInteger rmRequestId = new AtomicInteger();
  
  public ResourceManagerHandler(Configuration conf, ApplicationAttemptId appAttemptId) {
    this.conf = conf;
    this.appAttemptId = appAttemptId;
  }
  
  public AMRMProtocol getAMResourceManager() {
    if (amResourceManager != null)
      return amResourceManager;
    
    LOG.debug("Using configuration: " + conf);
    
    YarnConfiguration yarnConf = new YarnConfiguration(conf);
    YarnRPC rpc = YarnRPC.create(yarnConf);
    InetSocketAddress rmAddress = NetUtils.createSocketAddr(yarnConf.get(
        YarnConfiguration.RM_SCHEDULER_ADDRESS,
        YarnConfiguration.DEFAULT_RM_SCHEDULER_ADDRESS));

    LOG.info("Connecting to the resource manager (scheduling) at " + rmAddress);
    amResourceManager = (AMRMProtocol) rpc.getProxy(AMRMProtocol.class,
        rmAddress, conf);
    
    return amResourceManager;
  }
  
  public ClientRMProtocol getClientResourceManager() {
    if (clientResourceManager != null)
      return clientResourceManager;
    
    YarnConfiguration yarnConf = new YarnConfiguration(conf);
    YarnRPC rpc = YarnRPC.create(yarnConf);
    InetSocketAddress rmAddress = NetUtils.createSocketAddr(yarnConf.get(
        YarnConfiguration.RM_ADDRESS,
        YarnConfiguration.DEFAULT_RM_ADDRESS));
    
    LOG.info("Connecting to the resource manager (client) at " + rmAddress);
    
    clientResourceManager = (ClientRMProtocol) rpc.getProxy(
        ClientRMProtocol.class, rmAddress, conf);
    
    return clientResourceManager;
  }
  
  public ApplicationId getApplicationId() throws YarnRemoteException  {
    if (clientResourceManager == null)
      throw new IllegalStateException(
          "Cannot get an application ID befire connecting to resource manager!");
    
    GetNewApplicationRequest appReq = Records.newRecord(GetNewApplicationRequest.class);
    GetNewApplicationResponse appRes = clientResourceManager.getNewApplication(appReq);
    LOG.info("Got a new application with id=" + appRes.getApplicationId());
    
    return appRes.getApplicationId();
  }
  
  public void submitApplication(ApplicationId appId, String appName, Map<String, String> env, 
      Map<String, LocalResource> localResources, 
      List<String> commands, int memory) throws URISyntaxException, IOException {
    
    if (clientResourceManager == null)
      throw new IllegalStateException(
          "Cannot submit an application without connecting to resource manager!");

    ApplicationSubmissionContext appCtx = Records.newRecord(ApplicationSubmissionContext.class);
    appCtx.setApplicationId(appId);
    appCtx.setApplicationName(appName);
    appCtx.setQueue("default");
    appCtx.setUser(UserGroupInformation.getCurrentUser().getShortUserName());
    
    //System.out.println( "Based on my current user I am: " + UserGroupInformation.getCurrentUser().getShortUserName() );
        
    Priority prio = Records.newRecord(Priority.class);
    prio.setPriority(0);
    appCtx.setPriority(prio);

    
    // Launch ctx
    ContainerLaunchContext containerCtx = Records.newRecord(ContainerLaunchContext.class);
    containerCtx.setLocalResources(localResources);
    containerCtx.setCommands(commands);
    containerCtx.setEnvironment(env);
    containerCtx.setUser(UserGroupInformation.getCurrentUser().getShortUserName());
    
    Resource capability = Records.newRecord(Resource.class);
    capability.setMemory(memory);
    containerCtx.setResource(capability);
    
    appCtx.setAMContainerSpec(containerCtx);

    SubmitApplicationRequest submitReq = Records.newRecord(SubmitApplicationRequest.class);
    submitReq.setApplicationSubmissionContext(appCtx);
    
    LOG.info("Submitting application to ASM");
    clientResourceManager.submitApplication(submitReq);
    

    
    // Don't return anything, ASM#submit returns an empty response
  }
  
  public ApplicationReport getApplicationReport(ApplicationId appId)
      throws YarnRemoteException {

    if (clientResourceManager == null)
      throw new IllegalStateException(
          "Cannot query for a report without first connecting!");

    GetApplicationReportRequest req = Records
        .newRecord(GetApplicationReportRequest.class);
    req.setApplicationId(appId);

    return clientResourceManager.getApplicationReport(req).getApplicationReport();
  }
  
  public List<NodeReport> getClusterNodes() throws YarnRemoteException {
    if (clientResourceManager == null)
      throw new IllegalArgumentException("Can't get report without connecting first!");
    
    GetClusterNodesRequest req = Records.newRecord(GetClusterNodesRequest.class);
    GetClusterNodesResponse res = clientResourceManager.getClusterNodes(req);
    
    return res.getNodeReports();
    
  }
  
  public RegisterApplicationMasterResponse registerApplicationMaster(String host, int port)
      throws YarnRemoteException {
    
    if (amResourceManager == null)
      throw new IllegalStateException(
          "Cannot register application master before connecting to the resource manager!");
    
    RegisterApplicationMasterRequest request = Records
        .newRecord(RegisterApplicationMasterRequest.class);
    
    request.setApplicationAttemptId(appAttemptId);
    request.setHost(host);
    request.setRpcPort(port);
    request.setTrackingUrl("http://some-place.com/some/endpoint");
    
    LOG.info("Sending application registration request"
        + ", masterHost=" + request.getHost()
        + ", masterRpcPort=" + request.getRpcPort()
        + ", trackingUrl=" + request.getTrackingUrl()
        + ", applicationAttempt=" + request.getApplicationAttemptId()
        + ", applicationId=" + request.getApplicationAttemptId().getApplicationId());


    RegisterApplicationMasterResponse response = amResourceManager.registerApplicationMaster(request);
    LOG.debug("Received a registration response"
        + ", min=" + response.getMinimumResourceCapability().getMemory()
        + ", max=" + response.getMaximumResourceCapability().getMemory());
    
    return response;
  }

  /**
   * Changed the return type to AllocateResponse which use to hold a reference to 
   * AMResponse. 
   * 
   * AMResponse seems to have disappeared in CDH 4.6
   * 
   * @param requestedContainers
   * @param releasedContainers
   * @return
   * @throws YarnRemoteException
   */
  
  public AllocateResponse allocateRequest (
	      List<ResourceRequest> requestedContainers,
	      List<ContainerId> releasedContainers) throws YarnRemoteException {
	    
	    if (amResourceManager == null)
	      throw new IllegalStateException(
	          "Cannot send allocation request before connecting to the resource manager!");

	    LOG.info("Sending allocation request"
	        + ", requestedSize=" + requestedContainers.size()
	        + ", releasedSize=" + releasedContainers.size());
	    
	    for (ResourceRequest req : requestedContainers)
	      LOG.info("Requesting container, host=" + req.getHostName() 
	          + ", amount=" + req.getNumContainers()
	          + ", memory=" + req.getCapability().getMemory()
	          + ", priority=" + req.getPriority().getPriority());
	    
	    for (ContainerId rel : releasedContainers)
	      LOG.info("Releasing container: " + rel.getId());
	    
	    AllocateRequest request = Records.newRecord(AllocateRequest.class);
	    request.setResponseId(rmRequestId.incrementAndGet());
	    request.setApplicationAttemptId(appAttemptId);
	    request.addAllAsks(requestedContainers);
	    request.addAllReleases(releasedContainers);

	    AllocateResponse response = amResourceManager.allocate(request);
	    
	    //response.getAllocatedContainers()
	    
	    LOG.debug("Got an allocation response, "
	        + ", responseId=" + response.getResponseId()
	        + ", numClusterNodes=" + response.getNumClusterNodes()
	        + ", headroom=" + response.getAvailableResources().getMemory()
	        + ", allocatedSize=" + response.getAllocatedContainers().size()
	        + ", updatedNodes=" + response.getUpdatedNodes().size()
	        + ", reboot=" + response.getReboot()
	        + ", completedSize=" + response.getCompletedContainersStatuses().size());
	    
	    return response;
	  }  
  
  public void finishApplication(String diagnostics,
      FinalApplicationStatus finishState) throws YarnRemoteException {
    
    if (amResourceManager == null)
      throw new IllegalStateException(
          "Cannot finish an application without connecting to resource manager!");

    FinishApplicationMasterRequest request = Records.newRecord(FinishApplicationMasterRequest.class);
    request.setAppAttemptId(appAttemptId);
    request.setDiagnostics(diagnostics);
    request.setFinishApplicationStatus(finishState);

    LOG.info("Sending finish application notification "
        + ", state=" + request.getFinalApplicationStatus()
        + ", diagnostics=" + request.getDiagnostics());
    
    amResourceManager.finishApplicationMaster(request);
  }
}
