package org.deeplearning4j.aws.ec2;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.aws.s3.BaseS3;

import com.amazonaws.services.ec2.model.LaunchSpecification;
import com.amazonaws.services.ec2.model.RequestSpotInstancesRequest;
import com.amazonaws.services.ec2.model.RequestSpotInstancesResult;
import com.amazonaws.services.ec2.model.RunInstancesRequest;
import com.amazonaws.services.ec2.model.SpotInstanceRequest;
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


	public void createSpot() {
		// Initializes a Spot Instance Request
		RequestSpotInstancesRequest requestRequest = new RequestSpotInstancesRequest();

		// Request 1 x t1.micro instance with a bid price of $0.03.
		requestRequest.setSpotPrice("0.03");
		requestRequest.setInstanceCount(Integer.valueOf(1));

		// Setup the specifications of the launch. This includes the
		// instance type (e.g. t1.micro) and the latest Amazon Linux
		// AMI id available. Note, you should always use the latest
		// Amazon Linux AMI id or another of your choosing.
		LaunchSpecification launchSpecification = new LaunchSpecification();
		launchSpecification.setImageId("ami-8c1fece5");
		launchSpecification.setInstanceType("t1.micro");

		// Add the security group to the request.
		List<String> securityGroups = new ArrayList<String>();
		securityGroups.add("GettingStartedGroup");
		launchSpecification.setSecurityGroups(securityGroups);

		// Add the launch specifications to the request.
		requestRequest.setLaunchSpecification(launchSpecification);

		// Call the RequestSpotInstance API.
		RequestSpotInstancesResult requestResult = getEc2().requestSpotInstances(requestRequest);
		
		
		List<SpotInstanceRequest> requestResponses = requestResult.getSpotInstanceRequests();

		// Setup an arraylist to collect all of the request ids we want to
		// watch hit the running state.
		List<String> spotInstanceRequestIds = new ArrayList<String>();

		// Add all of the request ids to the hashset, so we can determine when they hit the
		// active state.
		for (SpotInstanceRequest requestResponse : requestResponses) {
		    System.out.println("Created Spot Request: "+requestResponse.getSpotInstanceRequestId());
		    spotInstanceRequestIds.add(requestResponse.getSpotInstanceRequestId());
		}
		
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
