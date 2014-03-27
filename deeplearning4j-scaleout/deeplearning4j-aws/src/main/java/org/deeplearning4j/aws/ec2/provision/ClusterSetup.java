package org.deeplearning4j.aws.ec2.provision;

import java.io.File;
import java.util.List;

import org.deeplearning4j.aws.ec2.Ec2BoxCreator;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sets up a DL4J cluster
 * @author Adam Gibson
 *
 */
public class ClusterSetup {

	@Option(name = "-w",usage = "Number of workers")
	private int numWorkers = 1;
	@Option(name = "-ami",usage = "Amazon machine image: default, amazon linux (only works with RHEL right now")
	private String ami = "ami-bba18dd2";
	@Option(name = "-s",usage = "size of instance: default m1.medium")
	private String size = "m1.medium";
	@Option(name = "-sg",usage = "security group, this needs to be set")
	private String securityGroupName;
	@Option(name = "-kp",usage = "key pair name, also needs to be set.")
	private String keyPairName;
	@Option(name = "-kpath",usage = "path to private key - needs to be set, this is used to login to amazon.")
	private String pathToPrivateKey;

	@Option(name = "-wscript", usage = "path to worker script to run, this will allow customization of dependencies")
	private String workerSetupScriptPath;
	@Option(name = "-mscript", usage = "path to master script to run this will allow customization of the dependencies")
	private String masterSetupScriptPath;
	@Option(name = "-lib",usage = "path to lib directory, this could be a default dl4j distribution or your own custom dependencies")
	private String libDirPath;
	@Option(name = "-datapath", usage = "path to serialized dataset")
	private String dataSetPath;
	@Option(name = "-uploddeps",usage = "whether to uploade deps: default true")
	private boolean uploadDeps = true;
	
	private static Logger log = LoggerFactory.getLogger(ClusterSetup.class);


	public ClusterSetup(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			parser.printUsage(System.err);
			log.error("Unable to parse args",e);
		}


	}

	public void exec() {
		//master + workers
		Ec2BoxCreator boxCreator = new Ec2BoxCreator(ami,numWorkers + 1,size,securityGroupName,keyPairName);
		boxCreator.create();
		boxCreator.blockTillAllRunning();
		List<String> hosts = boxCreator.getHosts();
		provisionMaster(hosts.get(0));
		//provisionWorkers(hosts.subList(1, hosts.size()));


	}




	private void provisionMaster(String host) {
		try {
			HostProvisioner uploader = new HostProvisioner(host, "ec2-user");
			uploader.addKeyFile(pathToPrivateKey);

			uploader.uploadForDeployment(libDirPath, "lib");
			if(dataSetPath != null)
				uploader.uploadForDeployment(dataSetPath, "");
			String sshPath = "/home/ec2-user/.ssh/";
			uploader.uploadForDeployment(pathToPrivateKey, sshPath + new File(pathToPrivateKey).getName());
			uploader.runRemoteCommand("chmod 0400 " + sshPath + "*");

			uploader.uploadAndRun(masterSetupScriptPath, "");
		}catch(Exception e) {
			log.error("Error ",e);
		}
	}

	private void provisionWorkers(List<String> workers) {
		for(String workerHost : workers) {
			try {
				HostProvisioner uploader = new HostProvisioner(workerHost, "ec2-user");
				uploader.addKeyFile(pathToPrivateKey);

				uploader.uploadForDeployment(libDirPath, "lib");
				uploader.uploadAndRun(workerSetupScriptPath, "");
			}catch(Exception e) {
				log.error("Error ",e);
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
