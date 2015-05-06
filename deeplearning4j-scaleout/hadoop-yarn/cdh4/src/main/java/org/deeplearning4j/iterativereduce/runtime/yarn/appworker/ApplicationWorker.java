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

package org.deeplearning4j.iterativereduce.runtime.yarn.appworker;


import org.apache.hadoop.mapreduce.RecordReader;
import org.deeplearning4j.iterativereduce.impl.reader.CanovaRecordReader;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.scaleout.api.ir.Updateable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetSocketAddress;

/*
 * Future YARN entry point
 */
public class ApplicationWorker<T extends Updateable> extends
    Configured implements Tool {

  private static final Logger LOG = LoggerFactory.getLogger(ApplicationWorker.class);
  
  protected CanovaRecordReader parser;
  protected ComputableWorker<T> computable;
  protected Class<T> updateable;

  public ApplicationWorker(CanovaRecordReader parser,
      ComputableWorker<T> computeable, Class<T> updateable) {

    this.parser = parser;
    this.computable = computeable;
    this.updateable = updateable;
  }

  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 4 || !args[0].equals("--master-addr") || !args[2].equals("--worker-id"))
      throw new IllegalArgumentException(
          "Expected two and only two options: --master-addr <host:port> and --worker-id <workerid>");

    String[] masterHostPort = args[1].split(":");
    InetSocketAddress masterAddr = new InetSocketAddress(masterHostPort[0],
        Integer.parseInt(masterHostPort[1]));
    Configuration conf = getConf();
    ApplicationWorkerService<T> worker = new ApplicationWorkerService(
        args[3], masterAddr, parser, computable, updateable, conf);

    LOG.info("Starting worker"
        + ", workerId=" + args[3]
        + ", masterHost=" + args[1]
        + ", parser=" + parser.getClass().getName()
        + ", computable=" + computable.getClass().getName()
        + ", updateable=" + updateable.getName());
    
    // Launch, and wait for completion
    int rc = worker.run();
    LOG.info("Worker completed with exit code " + rc);
    
    return rc;
  }
}