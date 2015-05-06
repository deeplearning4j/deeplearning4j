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

package org.deeplearning4j.iterativereduce.runtime.yarn.client;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.ClientRMProtocol;
import org.apache.hadoop.yarn.api.protocolrecords.KillApplicationRequest;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;
import org.deeplearning4j.iterativereduce.runtime.yarn.ResourceManagerHandler;

public class Kill {
  public static Log LOG = LogFactory.getLog(Kill.class);
  
  public static void main(String[] args) throws Exception {
    if (args.length < 1)
      throw new IllegalArgumentException("Need at least one argument - appId to kill");

    ApplicationId appId = ConverterUtils.toApplicationId(args[0]);
    LOG.info("Using Application ID: " + appId.toString());
    
    ResourceManagerHandler rmh = new ResourceManagerHandler(new Configuration(), null);
    ClientRMProtocol crm = rmh.getClientResourceManager();
    
    KillApplicationRequest kar = Records.newRecord(KillApplicationRequest.class);
    kar.setApplicationId(appId);
    
    LOG.info("Sending kill request");
    crm.forceKillApplication(kar);
  }
}
