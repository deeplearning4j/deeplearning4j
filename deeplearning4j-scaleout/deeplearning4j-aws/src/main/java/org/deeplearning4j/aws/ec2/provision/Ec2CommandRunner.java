package org.deeplearning4j.aws.ec2.provision;

import java.io.File;
import java.util.List;

import org.deeplearning4j.aws.ec2.Ec2BoxCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Ec2CommandRunner {

	private static Logger log = LoggerFactory.getLogger(Ec2CommandRunner.class);
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		Ec2BoxCreator boxCreator = new Ec2BoxCreator("ami-bba18dd2",1, "m1.medium","sg-61963904","blix");
		boxCreator.create();
		boxCreator.blockTillAllRunning();
		List<String> hosts = boxCreator.getHosts();
		Thread.sleep(15000);
		try {
			HostProvisioner uploader = new HostProvisioner(hosts.get(0), "ec2-user");
			String pathToPrivateKey = "/home/agibsonccc/.ssh/id_rsa";
			uploader.addKeyFile(pathToPrivateKey);
			String sshPath = "/home/ec2-user/.ssh/";
			uploader.uploadForDeployment(pathToPrivateKey, sshPath + new File(pathToPrivateKey).getName());
			uploader.runRemoteCommand("chmod 0400 " + sshPath + "*");
			uploader.uploadAndRun("/home/agibsonccc/test.sh", "");
		}catch(Exception e) {
			log.error("Error ",e);
		}
		
		boxCreator.blowupBoxes();
		
	}

}
