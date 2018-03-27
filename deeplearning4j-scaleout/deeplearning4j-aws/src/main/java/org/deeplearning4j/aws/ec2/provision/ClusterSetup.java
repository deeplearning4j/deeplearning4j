/*-
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

package org.deeplearning4j.aws.ec2.provision;

import com.amazonaws.regions.Regions;
import org.deeplearning4j.aws.ec2.Ec2BoxCreator;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.threadly.concurrent.PriorityScheduler;

import java.util.List;

/**
 * Sets up a DL4J cluster
 * @author Adam Gibson
 *
 */
public class ClusterSetup {

    @Option(name = "-w", usage = "Number of workers")
    private int numWorkers = 1;
    @Option(name = "-ami", usage = "Amazon machine image: default, amazon linux (only works with RHEL right now")
    private String ami = "ami-fb8e9292";
    @Option(name = "-s", usage = "size of instance: default m1.medium")
    private String size = "m3.xlarge";
    @Option(name = "-sg", usage = "security group, this needs to be applyTransformToDestination")
    private String securityGroupName;
    @Option(name = "-kp", usage = "key pair name, also needs to be applyTransformToDestination.")
    private String keyPairName;
    @Option(name = "-kpath",
                    usage = "path to private key - needs to be applyTransformToDestination, this is used to login to amazon.")
    private String pathToPrivateKey;
    @Option(name = "-wscript", usage = "path to worker script to run, this will allow customization of dependencies")
    private String workerSetupScriptPath;
    @Option(name = "-mscript", usage = "path to master script to run this will allow customization of the dependencies")
    private String masterSetupScriptPath;
    @Option(name = "-region", usage = "specify a region")
    private String region = Regions.US_EAST_1.getName();

    private PriorityScheduler as;

    private static final Logger log = LoggerFactory.getLogger(ClusterSetup.class);


    public ClusterSetup(String[] args) {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            parser.printUsage(System.err);
            log.error("Unable to parse args", e);
        }


    }

    public void exec() {
        //master + workers
        Ec2BoxCreator boxCreator = new Ec2BoxCreator(ami, numWorkers, size, securityGroupName, keyPairName);
        boxCreator.setRegion(Regions.fromName(region));
        boxCreator.create();
        boxCreator.blockTillAllRunning();
        List<String> hosts = boxCreator.getHosts();
        //provisionMaster(hosts.get(0));
        provisionWorkers(hosts);


    }



    private void provisionWorkers(List<String> workers) {
        as = new PriorityScheduler(Runtime.getRuntime().availableProcessors());
        for (final String workerHost : workers) {
            try {
                as.execute(new Runnable() {
                    @Override
                    public void run() {
                        HostProvisioner uploader = new HostProvisioner(workerHost, "ec2-user");
                        try {
                            uploader.addKeyFile(pathToPrivateKey);
                            //uploader.runRemoteCommand("sudo hostname " + workerHost);
                            uploader.uploadAndRun(workerSetupScriptPath, "");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                    }
                });

            } catch (Exception e) {
                log.error("Error ", e);
            }
        }
    }


    /**
     * @param args
     */
    public static void main(String[] args) {
        new ClusterSetup(args).exec();
    }

}
