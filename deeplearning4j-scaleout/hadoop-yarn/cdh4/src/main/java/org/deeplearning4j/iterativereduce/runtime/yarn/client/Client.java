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


import org.deeplearning4j.iterativereduce.runtime.ConfigFields;
import org.deeplearning4j.iterativereduce.runtime.Utils;
import org.deeplearning4j.iterativereduce.runtime.yarn.ResourceManagerHandler;
import org.apache.commons.lang.time.StopWatch;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.yarn.api.records.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public class Client extends Configured implements Tool {

  private static final Logger LOG = LoggerFactory.getLogger(Client.class);
  
  /**
   * TODO: consider the scenarios where we dont get enough containers 
   * - we need to re-submit the job till we get the containers alloc'd
   * 
   */
  @Override
  public int run(String[] args) throws Exception {
	  
	  //System.out.println("IR: Client.run() [start]");
	  
    if (args.length < 1)
      LOG.info("No configuration file specified, using default ("
          + ConfigFields.DEFAULT_CONFIG_FILE + ")");
    
    long startTime = System.currentTimeMillis();
    String configFile = (args.length < 1) ? ConfigFields.DEFAULT_CONFIG_FILE : args[0];
    Properties props = new Properties();
    Configuration conf = getConf();

    try {
      FileInputStream fis = new FileInputStream(configFile);
      props.load(fis);
    } catch (FileNotFoundException ex) {
      throw ex; // TODO: be nice
    } catch (IOException ex) {
      throw ex; // TODO: be nice
    }
    
    // Make sure we have some bare minimums
    ConfigFields.validateConfig(props);
    
    if (LOG.isDebugEnabled()) {
      LOG.debug("Loaded configuration: ");
      for (Map.Entry<Object, Object> entry : props.entrySet()) {
        LOG.debug(entry.getKey() + "=" + entry.getValue());
      }
    }

    // TODO: make sure input file(s), libs, etc. actually exist!
    // Ensure our input path exists
    
    Path p = new Path(props.getProperty(ConfigFields.APP_INPUT_PATH));
    FileSystem fs = FileSystem.get(conf);
    
    if (!fs.exists(p))
      throw new FileNotFoundException("Input path not found: " + p.toString()
          + " (in " + fs.getUri() + ")");

    LOG.info("Using input path: " + p.toString());
    
    // Connect
    ResourceManagerHandler rmHandler = new ResourceManagerHandler(conf, null);
    rmHandler.getClientResourceManager();

    // Create an Application request/ID
    ApplicationId appId = rmHandler.getApplicationId(); // Our AppId
    String appName = props.getProperty(ConfigFields.APP_NAME,
        ConfigFields.DEFAULT_APP_NAME).replace(' ', '_');
    
    LOG.info("Got an application, id=" + appId + ", appName=" + appName);
    
    // Copy resources to [HD]FS
    LOG.debug("Copying resources to filesystem");
    Utils.copyLocalResourcesToFs(props, conf, appId, appName); // Local resources
    Utils.copyLocalResourceToFs(configFile, ConfigFields.APP_CONFIG_FILE, conf,
        appId, appName); // Config file
    
    try {
      Utils.copyLocalResourceToFs("log4j.properties", "log4j.properties", conf,
          appId, appName); // Log4j
    } catch (FileNotFoundException ex) {
      LOG.warn("log4j.properties file not found");
    }

    // Create our context
    List<String> commands = Utils.getMasterCommand(conf, props);
    Map<String, LocalResource> localResources = Utils
        .getLocalResourcesForApplication(conf, appId, appName, props,
            LocalResourceVisibility.APPLICATION);

    // Submit app
    rmHandler.submitApplication(appId, appName, Utils.getEnvironment(conf, props), 
        localResources, commands,
        Integer.parseInt(props.getProperty(ConfigFields.YARN_MEMORY, "512")));    

    /*
     * TODO:
     * - look at updating this code region to make sure job is submitted!
     * 
     */
    
	StopWatch watch = new StopWatch();
	watch.start();
    
    
    // Wait for app to complete
    while (true) {
      Thread.sleep(2000);
      
      ApplicationReport report = rmHandler.getApplicationReport(appId);
      LOG.info("IterativeReduce report: "
          + " appId=" + appId.getId()
          + ", state: " + report.getYarnApplicationState().toString()
          + ", Running Time: " + watch.toString() );

      
      //report.getDiagnostics()

      if (YarnApplicationState.FINISHED == report.getYarnApplicationState()) {
        LOG.info("Application finished in " + (System.currentTimeMillis() - startTime) + "ms");

        if (FinalApplicationStatus.SUCCEEDED == report.getFinalApplicationStatus()) {
          LOG.info("Application completed succesfully.");
          return 0;
        } else {
          LOG.info("Application completed with en error: " + report.getDiagnostics());
          return -1;
        }
      } else if (YarnApplicationState.FAILED == report.getYarnApplicationState() ||
          YarnApplicationState.KILLED == report.getYarnApplicationState()) {

        LOG.info("Application completed with a failed or killed state: " + report.getDiagnostics());
        return -1; 
      }
      
      
      
    }
    
    
    
    
    
  }
  
  public static void main(String[] args) throws Exception {
    int rc = ToolRunner.run(new Configuration(), new Client(), args);
    
    // Log, because been bitten before on daemon threads; sanity check
    LOG.debug("Calling System.exit(" + rc + ")");
    System.exit(rc);
  }
}
