package org.deeplearning4j.aws.ec2;

import org.deeplearning4j.aws.s3.BaseS3;

import com.amazonaws.services.ec2.model.RunInstancesRequest;
/**
 * Creates Ec2Boxes
 * @author Adam Gibson
 *
 */
public class Ec2BoxCreator extends BaseS3 {


	private String amiId;
	private int numBoxes;
	private String size;
	
	/**
	 * 
	 * @param amiId amazon image id
	 * @param numBoxes number of boxes
	 * @param size the size of the instances
	 */
	public Ec2BoxCreator(String amiId, int numBoxes,String size) {
		super();
		this.amiId = amiId;
		this.numBoxes = numBoxes;
		this.size = size;
	}


	public void create() {
		RunInstancesRequest runInstancesRequest = 
				new RunInstancesRequest();

		runInstancesRequest.withImageId(amiId)
		.withInstanceType(size)
		.withMinCount(1)
		.withMaxCount(numBoxes);
		getEc2().runInstances(runInstancesRequest);
	}

}
