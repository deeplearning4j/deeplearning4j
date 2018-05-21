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

package org.deeplearning4j.aws.ec2;

import com.amazonaws.regions.Regions;
import com.amazonaws.services.ec2.AmazonEC2;
import com.amazonaws.services.ec2.model.*;
import org.deeplearning4j.aws.s3.BaseS3;
import org.deeplearning4j.util.ThreadUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Creates Ec2Boxes
 * @author Adam Gibson
 *
 */
public class Ec2BoxCreator extends BaseS3 {


    private String amiId;
    private int numBoxes;
    private String size;
    private List<String> boxesCreated;
    private String securityGroupId;
    private String keyPair;
    private Regions regions = Regions.DEFAULT_REGION;
    private static final Logger log = LoggerFactory.getLogger(Ec2BoxCreator.class);

    //centos
    public final static String DEFAULT_AMI = "ami-8997afe0";

    /**
     * 
     * @param numBoxes number of boxes
     * @param size the size of the instances
     */
    public Ec2BoxCreator(int numBoxes, String size, String securityGroupId, String keyPair) {
        this(DEFAULT_AMI, numBoxes, size, securityGroupId, keyPair);
    }


    /**
     * 
     * @param amiId amazon image id
     * @param numBoxes number of boxes
     * @param size the size of the instances
     * @param securityGroupId
     */
    public Ec2BoxCreator(String amiId, int numBoxes, String size, String securityGroupId, String keyPair) {
        super();
        this.amiId = amiId;
        this.numBoxes = numBoxes;
        this.size = size;
        this.keyPair = keyPair;
        this.securityGroupId = securityGroupId;
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
        List<String> securityGroups = new ArrayList<>();
        securityGroups.add("GettingStartedGroup");
        launchSpecification.setSecurityGroups(securityGroups);

        // Add the launch specifications to the request.
        requestRequest.setLaunchSpecification(launchSpecification);

        // Call the RequestSpotInstance API.
        RequestSpotInstancesResult requestResult = getEc2().requestSpotInstances(requestRequest);


        List<SpotInstanceRequest> requestResponses = requestResult.getSpotInstanceRequests();

        // Setup an arraylist to collect all of the request ids we want to
        // watch hit the running state.
        List<String> spotInstanceRequestIds = new ArrayList<>();

        // Add all of the request ids to the hashset, so we can determine when they hit the
        // active state.
        for (SpotInstanceRequest requestResponse : requestResponses) {
            System.out.println("Created Spot Request: " + requestResponse.getSpotInstanceRequestId());
            spotInstanceRequestIds.add(requestResponse.getSpotInstanceRequestId());
        }

    }

    public void setRegion(Regions regions) {
        this.regions = regions;
    }


    /**
     * Create the instances
     */
    public void create() {
        RunInstancesRequest runInstancesRequest =
                        new RunInstancesRequest().withImageId(amiId).withInstanceType(size).withKeyName(keyPair)
                                        .withMinCount(1).withSecurityGroupIds(securityGroupId).withMaxCount(numBoxes);
        AmazonEC2 ec2 = getEc2();
        ec2.setRegion(com.amazonaws.regions.Region.getRegion(regions));
        List<Instance> boxes = ec2.runInstances(runInstancesRequest).getReservation().getInstances();
        if (boxesCreated == null) {
            boxesCreated = new ArrayList<>();
            for (Instance i : boxes)
                boxesCreated.add(i.getInstanceId());



            log.info("Boxes created " + boxesCreated);
        } else {
            blowupBoxes();
            boxesCreated.clear();
            for (Instance i : boxes)
                boxesCreated.add(i.getInstanceId());

        }
    }



    public List<InstanceStateChange> blowupBoxes() {
        TerminateInstancesRequest request = new TerminateInstancesRequest().withInstanceIds(boxesCreated);

        if (boxesCreated != null) {
            TerminateInstancesResult result = getEc2().terminateInstances(request);
            List<InstanceStateChange> change = result.getTerminatingInstances();
            log.info("Boxes destroyed " + boxesCreated);
            return change;
        }

        return Collections.emptyList();
    }


    public void blockTillAllRunning() {
        while (!allRunning()) {
            ThreadUtils.uncheckedSleep(1000);
            log.info("Not all created...");
        }
    }

    public boolean allRunning() {
        if (boxesCreated == null)
            return false;
        else {
            DescribeInstancesResult result = getEc2().describeInstances();
            List<Reservation> reservations = result.getReservations();
            for (Reservation r : reservations) {
                List<Instance> instances = r.getInstances();
                for (Instance instance : instances) {
                    InstanceState state = instance.getState();
                    if (state.getCode() == 48)
                        continue;
                    if (state.getCode() != 16)
                        return false;
                }
            }

            return true;
        }


    }

    public List<String> getHosts() {
        DescribeInstancesResult result = getEc2().describeInstances();
        List<String> hosts = new ArrayList<>();
        List<Reservation> reservations = result.getReservations();
        for (Reservation r : reservations) {
            List<Instance> instances = r.getInstances();
            for (Instance instance : instances) {
                InstanceState state = instance.getState();
                if (state.getCode() == 48)
                    continue;
                hosts.add(instance.getPublicDnsName());

            }
        }

        return hosts;
    }

    public List<String> getBoxesCreated() {
        return boxesCreated;
    }



}
