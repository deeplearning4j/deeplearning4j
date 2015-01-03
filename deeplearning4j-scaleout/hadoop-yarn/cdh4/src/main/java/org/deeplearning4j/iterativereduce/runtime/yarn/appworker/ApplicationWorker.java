package org.deeplearning4j.iterativereduce.runtime.yarn.appworker;


import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.iterativereduce.runtime.Updateable;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.deeplearning4j.iterativereduce.runtime.io.RecordParser;

import java.net.InetSocketAddress;

/*
 * Future YARN entry point
 */
public class ApplicationWorker<T extends Updateable> extends
    Configured implements Tool {

  private static final Log LOG = LogFactory.getLog(ApplicationWorker.class);
  
  protected RecordParser<T> parser;
  protected ComputableWorker<T> computable;
  protected Class<T> updateable;

  public ApplicationWorker(RecordParser<T> parser,
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
    ApplicationWorkerService<T> worker = new ApplicationWorkerService<T>(
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