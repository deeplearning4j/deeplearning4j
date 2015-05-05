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
import org.apache.hadoop.yarn.api.ContainerManager;
import org.apache.hadoop.yarn.api.protocolrecords.GetContainerStatusRequest;
import org.apache.hadoop.yarn.api.protocolrecords.GetContainerStatusResponse;
import org.apache.hadoop.yarn.api.protocolrecords.StartContainerRequest;
import org.apache.hadoop.yarn.api.protocolrecords.StartContainerResponse;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnRemoteException;
import org.apache.hadoop.yarn.ipc.YarnRPC;
import org.apache.hadoop.yarn.util.Records;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.Map;

public class ContainerManagerHandler {

  private static final Logger LOG = LoggerFactory.getLogger(ContainerManagerHandler.class);
  
  private Configuration conf;
  private Container container;
  private ContainerManager containerManager;
  
  public ContainerManagerHandler (Configuration conf, Container container) {
    this.conf = conf;
    this.container = container;
  }
  
  public ContainerManager getContainerManager() {
    if (containerManager != null)
      return containerManager;
    
    YarnConfiguration yarnConf = new YarnConfiguration(conf);
    YarnRPC rpc = YarnRPC.create(yarnConf);
    
    InetSocketAddress cmAddr = NetUtils.createSocketAddr(container.getNodeId()
        .getHost(), container.getNodeId().getPort());
    
    LOG.info("Connecting to container manager at " + cmAddr);
    containerManager = ((ContainerManager) rpc.getProxy(ContainerManager.class,
        cmAddr, conf));
   
    return containerManager;
  }
  
  public StartContainerResponse startContainer(List<String> commands,
      Map<String, LocalResource> localResources, Map<String, String> env) throws IOException {

    if (containerManager == null)
      throw new IllegalStateException(
          "Cannot start a continer before connecting to the container manager!");

    ContainerLaunchContext ctx = Records.newRecord(ContainerLaunchContext.class);
    ctx.setContainerId(container.getId());
    ctx.setResource(container.getResource());
    ctx.setLocalResources(localResources);
    ctx.setCommands(commands);
    ctx.setUser(UserGroupInformation.getCurrentUser().getShortUserName());
    ctx.setEnvironment(env);

    if (LOG.isDebugEnabled()) {
      LOG.debug("Using ContainerLaunchContext with"
          + ", containerId=" + ctx.getContainerId()
          + ", memory=" + ctx.getResource().getMemory()
          + ", localResources=" + ctx.getLocalResources().toString()
          + ", commands=" + ctx.getCommands().toString()
          + ", env=" + ctx.getEnvironment().toString());
    }
    
    StartContainerRequest request = Records.newRecord(StartContainerRequest.class);
    request.setContainerLaunchContext(ctx);
    
    LOG.info("Starting container, containerId=" + container.getId().toString()
        + ", host=" + container.getNodeId().getHost()
        + ", http=" + container.getNodeHttpAddress());
    
    return containerManager.startContainer(request);
  }
  
  public GetContainerStatusResponse getContainerStatus()
      throws YarnRemoteException {

    if (containerManager == null)
      throw new IllegalStateException(
          "Cannot request container status without connecting to container manager!");

    GetContainerStatusRequest request = Records.newRecord(GetContainerStatusRequest.class);
    request.setContainerId(container.getId());
    
    LOG.info("Getting container status, containerId=" + container.getId().toString());
    return containerManager.getContainerStatus(request);
  }
}