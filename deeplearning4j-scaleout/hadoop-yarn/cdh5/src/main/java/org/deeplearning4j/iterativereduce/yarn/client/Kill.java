package org.deeplearning4j.iterativereduce.yarn.client;

import com.cloudera.iterativereduce.yarn.ResourceManagerHandler;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.ClientRMProtocol;
import org.apache.hadoop.yarn.api.protocolrecords.KillApplicationRequest;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;

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
