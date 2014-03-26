package org.deeplearning4j.aws.ec2.provision;

import java.util.List;

import org.deeplearning4j.aws.ec2.Ec2BoxCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Ec2CommandRunner {

	private static Logger log = LoggerFactory.getLogger(Ec2CommandRunner.class);
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Ec2BoxCreator boxCreator = new Ec2BoxCreator(1, "m1.medium");
		boxCreator.create();
		boxCreator.blockTillAllRunning();
		List<String> hosts = boxCreator.getHosts();
		log.info("Hosts " + hosts);
		boxCreator.blowupBoxes();
		
	}

}
