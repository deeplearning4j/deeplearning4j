package org.deeplearning4j.aws.emr;

import com.amazonaws.services.elasticmapreduce.AmazonElasticMapReduce;
import com.amazonaws.services.elasticmapreduce.AmazonElasticMapReduceClientBuilder;
import com.amazonaws.services.elasticmapreduce.model.*;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.AmazonS3URI;
import com.amazonaws.services.s3.model.PutObjectRequest;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.RandomStringUtils;
import org.apache.spark.api.java.function.Function;

import java.io.File;
import java.util.*;

/**
 * Configuration for a Spark EMR cluster
 */
@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
@Slf4j
public class SparkEMRClient {

    protected String sparkClusterName = RandomStringUtils.randomAlphanumeric(12);
    protected String sparkAwsRegion = "us-east-1";
    protected String sparkEmrRelease = "emr-5.9.0";
    protected String sparkEmrServiceRole = "EMR_DefaultRole";
    protected List<EmrConfig> sparkEmrConfigs = Collections.emptyList();
    protected String sparkSubnetId = null;
    protected List<String> sparkSecurityGroupIds = Collections.emptyList();
    protected int sparkInstanceCount = 1;
    protected String sparkInstanceType = "m3.xlarge";
    protected Optional<Float> sparkInstanceBidPrice = Optional.empty();
    protected String sparkInstanceRole = "EMR_EC2_DefaultRole";
    protected String sparkS3JarFolder = "changeme";
    protected int sparkTimeoutDurationMinutes = 90;

    //underlying configs
    protected AmazonElasticMapReduceClientBuilder sparkEmrClientBuilder;
    protected AmazonS3ClientBuilder sparkS3ClientBuilder;
    protected JobFlowInstancesConfig sparkJobFlowInstancesConfig;
    protected RunJobFlowRequest sparkRunJobFlowRequest;
    protected Function<PutObjectRequest, PutObjectRequest> sparkS3PutObjectDecorator;
    protected Map<String, String> sparkSubmitConfs;


    private static ClusterState[] activeClusterStates = new ClusterState[]{
            ClusterState.RUNNING,
            ClusterState.STARTING,
            ClusterState.WAITING,
            ClusterState.BOOTSTRAPPING};

    private Optional<ClusterSummary> findClusterWithName(AmazonElasticMapReduce emr, String name) {
        List<ClusterSummary> csrl = emr.listClusters((new ListClustersRequest()).withClusterStates(activeClusterStates)).getClusters();
        for (ClusterSummary csr : csrl) {
            if (csr.getName().equals(name)) return Optional.of(csr);
        }
        return Optional.empty();
    }

    /**
     * Creates the current cluster
     */
    public void createCluster() {
        AmazonElasticMapReduce emr = sparkEmrClientBuilder.build();
        Optional<ClusterSummary> csr = findClusterWithName(emr, sparkClusterName);
        if (csr.isPresent()) {
            String msg = String.format("A cluster with name %s and id %s is already deployed", sparkClusterName, csr.get().getId());
            log.error(msg);
            throw new IllegalStateException(msg);
        } else {
            RunJobFlowResult res = emr.runJobFlow(sparkRunJobFlowRequest);
            String msg = String.format("Your cluster is launched with name %s and id %s.", sparkClusterName, res.getJobFlowId());
            log.info(msg);
        }

    }

    private void logClusters(List<ClusterSummary> csrl) {
        if (csrl.isEmpty()) log.info("No cluster found.");
        else {
            log.info(String.format("%d clusters found.", csrl.size()));
            for (ClusterSummary csr : csrl) {
                log.info(String.format("Name: %s | Id: %s", csr.getName(), csr.getId()));
            }
        }
    }

    /**
     * Lists existing active clusters Names
     *
     * @return cluster names
     */
    public List<String> listActiveClusterNames() {
        AmazonElasticMapReduce emr = sparkEmrClientBuilder.build();
        List<ClusterSummary> csrl =
                emr.listClusters(new ListClustersRequest().withClusterStates(activeClusterStates)).getClusters();
        logClusters(csrl);
        List<String> res = new ArrayList<>(csrl.size());
        for (ClusterSummary csr : csrl) res.add(csr.getName());
        return res;
    }

    /**
     * List existing active cluster IDs
     *
     * @return cluster IDs
     */
    public List<String> listActiveClusterIds() {
        AmazonElasticMapReduce emr = sparkEmrClientBuilder.build();
        List<ClusterSummary> csrl =
                emr.listClusters(new ListClustersRequest().withClusterStates(activeClusterStates)).getClusters();
        logClusters(csrl);
        List<String> res = new ArrayList<>(csrl.size());
        for (ClusterSummary csr : csrl) res.add(csr.getId());
        return res;
    }


    /**
     * Terminates a cluster
     */
    public void terminateCluster() {
        AmazonElasticMapReduce emr = sparkEmrClientBuilder.build();
        Optional<ClusterSummary> optClusterSum = findClusterWithName(emr, sparkClusterName);
        if (!optClusterSum.isPresent()) {
            log.error(String.format("The cluster with name %s , requested for deletion, does not exist.", sparkClusterName));
        } else {
            String id = optClusterSum.get().getId();
            emr.terminateJobFlows((new TerminateJobFlowsRequest()).withJobFlowIds(id));
            log.info(String.format("The cluster with id %s is terminating.", id));
        }
    }

    // The actual job-sumission logic
    private void submitJob(AmazonElasticMapReduce emr, String mainClass, List<String> args, Map<String, String> sparkConfs, File uberJar) throws Exception {
        AmazonS3URI s3Jar = new AmazonS3URI(sparkS3JarFolder + "/" + uberJar.getName());
        log.info(String.format("Placing uberJar %s to %s", uberJar.getPath(), s3Jar.toString()));
        PutObjectRequest putRequest = sparkS3PutObjectDecorator.call(
                new PutObjectRequest(s3Jar.getBucket(), s3Jar.getKey(), uberJar)
        );
        sparkS3ClientBuilder.build().putObject(putRequest);
        // The order of these matters
        List<String> sparkSubmitArgs = Arrays.asList(
                "spark-submit",
                "--deploy-mode",
                "cluster",
                "--class",
                mainClass
        );
        for (Map.Entry<String, String> e : sparkConfs.entrySet()) {
            sparkSubmitArgs.add(String.format("--conf %s = %s ", e.getKey(), e.getValue()));
        }
        sparkSubmitArgs.add(s3Jar.toString());
        sparkSubmitArgs.addAll(args);
        StepConfig step = new StepConfig()
                .withActionOnFailure(ActionOnFailure.CONTINUE)
                .withName("Spark step")
                .withHadoopJarStep(
                        new HadoopJarStepConfig()
                                .withJar("command-runner.jar")
                                .withArgs(sparkSubmitArgs)
                );

        Optional<ClusterSummary> optCsr = findClusterWithName(emr, sparkClusterName);
        if (optCsr.isPresent()) {
            ClusterSummary csr = optCsr.get();
            emr.addJobFlowSteps(
                    new AddJobFlowStepsRequest()
                            .withJobFlowId(csr.getId())
                            .withSteps(step));
            log.info(
                    String.format("Your job is added to the cluster with id %s.", csr.getId())
            );
        } else {
            // If the cluster wasn't started, it's assumed ot be throwaway
            List<StepConfig> steps = sparkRunJobFlowRequest.getSteps();
            steps.add(step);
            RunJobFlowRequest jobFlowRequest = sparkRunJobFlowRequest
                    .withSteps(steps)
                    .withInstances(sparkJobFlowInstancesConfig.withKeepJobFlowAliveWhenNoSteps(false));

            RunJobFlowResult res = emr.runJobFlow(jobFlowRequest);
            log.info("Your new cluster's id is %s.", res.getJobFlowId());
        }

    }

    /**
     * Submit a Spark Job with a specified main class
     */
    public void sparkSubmitJobWithMain(String[] args, String mainClass, File uberJar) throws Exception {
        AmazonElasticMapReduce emr = sparkEmrClientBuilder.build();
        submitJob(emr, mainClass, Arrays.asList(args), sparkSubmitConfs, uberJar);
    }

    private void checkStatus(AmazonElasticMapReduce emr, String clusterId) throws InterruptedException {
        log.info(".");
        com.amazonaws.services.elasticmapreduce.model.Cluster dcr =
                emr.describeCluster((new DescribeClusterRequest()).withClusterId(clusterId)).getCluster();
        String state = dcr.getStatus().getState();
        long timeOutTime = System.currentTimeMillis() + ((long) sparkTimeoutDurationMinutes * 60 * 1000);

        Boolean activated = Arrays.asList(activeClusterStates).contains(ClusterState.fromValue(state));
        Boolean timedOut = System.currentTimeMillis() > timeOutTime;
        if (activated && timedOut) {
            emr.terminateJobFlows(
                    new TerminateJobFlowsRequest().withJobFlowIds(clusterId)
            );
            log.error("Timeout. Cluster terminated.");
        } else if (!activated) {
            Boolean hasAbnormalStep = false;
            StepSummary stepS = null;
            List<StepSummary> steps = emr.listSteps(new ListStepsRequest().withClusterId(clusterId)).getSteps();
            for (StepSummary step : steps) {
                if (step.getStatus().getState() != StepState.COMPLETED.toString()) {
                    hasAbnormalStep = true;
                    stepS = step;
                }
            }
            if (hasAbnormalStep && stepS != null)
                log.error(String.format("Cluster %s terminated with an abnormal step, name %s, id %s", clusterId, stepS.getName(), stepS.getId()));
            else
                log.info("Cluster %s terminated without error.", clusterId);
        } else {
            Thread.sleep(5000);
            checkStatus(emr, clusterId);
        }
    }

    /**
     * Monitor the cluster and terminates when it times out
     */
    public void sparkMonitor() throws InterruptedException {
        AmazonElasticMapReduce emr = sparkEmrClientBuilder.build();
        Optional<ClusterSummary> optCsr = findClusterWithName(emr, sparkClusterName);
        if (!optCsr.isPresent()) {
            log.error(String.format("The cluster with name %s does not exist.", sparkClusterName));
        } else {
            ClusterSummary csr = optCsr.get();
            log.info(String.format("found cluster with id %s, starting monitoring", csr.getId()));
            checkStatus(emr, csr.getId());
        }
    }

    @Data
    public static class Builder {

        protected String sparkClusterName = RandomStringUtils.randomAlphanumeric(12);
        protected String sparkAwsRegion = "us-east-1";
        protected String sparkEmrRelease = "emr-5.9.0";
        protected String sparkEmrServiceRole = "EMR_DefaultRole";
        protected List<EmrConfig> sparkEmrConfigs = Collections.emptyList();
        protected String sparkSubNetid = null;
        protected List<String> sparkSecurityGroupIds = Collections.emptyList();
        protected int sparkInstanceCount = 1;
        protected String sparkInstanceType = "m3.xlarge";
        protected Optional<Float> sparkInstanceBidPrice = Optional.empty();
        protected String sparkInstanceRole = "EMR_EC2_DefaultRole";
        protected String sparkS3JarFolder = "changeme";
        protected int sparkTimeoutDurationMinutes = 90;

        protected AmazonElasticMapReduceClientBuilder sparkEmrClientBuilder;
        protected AmazonS3ClientBuilder sparkS3ClientBuilder;
        protected JobFlowInstancesConfig sparkJobFlowInstancesConfig;
        protected RunJobFlowRequest sparkRunJobFlowRequest;

        // This should allow the user to decorate the put call to add metadata to the jar put command, such as security groups,
        protected Function<PutObjectRequest, PutObjectRequest> sparkS3PutObjectDecorator = new Function<PutObjectRequest, PutObjectRequest>() {
            @Override
            public PutObjectRequest call(PutObjectRequest putObjectRequest) throws Exception {
                return putObjectRequest;
            }
        };
        protected Map<String, String> sparkSubmitConfs;


        /**
         * Defines the EMR cluster's name
         *
         * @param clusterName the EMR cluster's name
         * @return an EMR cluster builder
         */
        public Builder clusterName(String clusterName) {
            this.sparkClusterName = clusterName;
            return this;
        }

        /**
         * Defines the EMR cluster's region
         * See https://docs.aws.amazon.com/general/latest/gr/rande.html
         *
         * @param region the EMR cluster's region
         * @return an EMR cluster builder
         */
        public Builder awsRegion(String region) {
            this.sparkAwsRegion = region;
            return this;
        }

        /**
         * Defines the EMR release version to be used in this cluster
         * uses a release label
         * See https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-4.2.0/emr-release-differences.html#emr-release-label
         *
         * @param releaseLabel the EMR release label
         * @return an EM cluster Builder
         */
        public Builder emrRelease(String releaseLabel) {
            this.sparkEmrRelease = releaseLabel;
            return this;
        }

        /**
         * Defines the IAM role to be assumed by the EMR service
         * <p>
         * https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-service.html
         *
         * @param serviceRole the service role
         * @return an EM cluster Builder
         */
        public Builder emrServiceRole(String serviceRole) {
            this.sparkEmrServiceRole = serviceRole;
            return this;
        }

        /**
         * A list of configuration parameters to apply to EMR instances.
         *
         * @param configs the EMR configurations to apply to this cluster
         * @return an EMR cluster builder
         */
        public Builder emrConfigs(List<EmrConfig> configs) {
            this.sparkEmrConfigs = configs;
            return this;
        }

        /**
         * The id of the EC2 subnet to be used for this Spark EMR service
         * see https://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_Subnets.html
         *
         * @param id the subnet ID
         * @return an EMR cluster builder
         */
        public Builder subnetId(String id) {
            this.sparkSubNetid = id;
            return this;
        }

        /**
         * The id of additional security groups this deployment should adopt for both master and slaves
         *
         * @param securityGroups
         * @return an EMR cluster builder
         */
        public Builder securityGroupIDs(List<String> securityGroups) {
            this.sparkSecurityGroupIds = securityGroups;
            return this;
        }

        /**
         * The number of instances this deployment should comprise of
         *
         * @param count the number of instances for this cluster
         * @rturn an EMR cluster buidler
         */
        public Builder instanceCount(int count) {
            this.sparkInstanceCount = count;
            return this;
        }

        /**
         * The type of instance this cluster should comprise of
         * See https://aws.amazon.com/ec2/instance-types/
         *
         * @param instanceType the type of instance for this cluster
         * @return an EMR cluster builder
         */
        public Builder instanceType(String instanceType) {
            this.sparkInstanceType = instanceType;
            return this;
        }

        /**
         * The optional bid value for this cluster's spot instances
         * see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-spot-instances-work.html
         * Uses the on-demand market if empty.
         *
         * @param optBid the Optional bid price for this cluster's instnces
         * @return an EMR cluster Builder
         */
        public Builder instanceBidPrice(Optional<Float> optBid) {
            this.sparkInstanceBidPrice = optBid;
            return this;
        }

        /**
         * The EC2 instance role that this cluster's instances should assume
         * see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html
         *
         * @param role the intended instance role
         * @return an EMR cluster builder
         */
        public Builder instanceRole(String role) {
            this.sparkInstanceRole = role;
            return this;
        }

        /**
         * the S3 folder in which to find the application jar
         *
         * @param jarfolder the S3 folder in which to find a jar
         * @return an EMR cluster builder
         */
        public Builder s3JarFolder(String jarfolder) {
            this.sparkS3JarFolder = jarfolder;
            return this;
        }

        /**
         * The timeout duration for this Spark EMR cluster, in minutes
         *
         * @param timeoutMinutes
         * @return an EMR cluster builder
         */
        public Builder sparkTimeOutDurationMinutes(int timeoutMinutes) {
            this.sparkTimeoutDurationMinutes = timeoutMinutes;
            return this;
        }

        /**
         * Creates an EMR Spark cluster deployment
         *
         * @return a SparkEMRClient
         */
        public SparkEMRClient build() {
            this.sparkEmrClientBuilder = AmazonElasticMapReduceClientBuilder.standard().withRegion(sparkAwsRegion);
            this.sparkS3ClientBuilder = AmazonS3ClientBuilder.standard().withRegion(sparkAwsRegion);
            // note this will be kept alive without steps, an arbitrary choice to avoid rapid test-teardown-restart cycles
            this.sparkJobFlowInstancesConfig = (new JobFlowInstancesConfig()).withKeepJobFlowAliveWhenNoSteps(true);
            if (this.sparkSubNetid != null)
                this.sparkJobFlowInstancesConfig = this.sparkJobFlowInstancesConfig.withEc2SubnetId(this.sparkSubNetid);
            if (!this.sparkSecurityGroupIds.isEmpty()) {
                this.sparkJobFlowInstancesConfig = this.sparkJobFlowInstancesConfig.withAdditionalMasterSecurityGroups(this.sparkSecurityGroupIds);
                this.sparkJobFlowInstancesConfig = this.sparkJobFlowInstancesConfig.withAdditionalSlaveSecurityGroups(this.sparkSecurityGroupIds);
            }

            InstanceGroupConfig masterConfig =
                    (new InstanceGroupConfig()).withInstanceCount(1).withInstanceRole(InstanceRoleType.MASTER).withInstanceType(sparkInstanceType);
            if (sparkInstanceBidPrice.isPresent()) {
                masterConfig = masterConfig.withMarket(MarketType.SPOT).withBidPrice(sparkInstanceBidPrice.get().toString());
            } else {
                masterConfig = masterConfig.withMarket(MarketType.ON_DEMAND);
            }

            int slaveCount = sparkInstanceCount - 1;
            InstanceGroupConfig slaveConfig =
                    (new InstanceGroupConfig()).withInstanceCount(slaveCount).withInstanceRole(InstanceRoleType.CORE).withInstanceRole(sparkInstanceType);
            if (sparkInstanceBidPrice.isPresent()) {
                slaveConfig = slaveConfig.withMarket(MarketType.SPOT).withBidPrice(sparkInstanceBidPrice.get().toString());
            } else {
                slaveConfig = slaveConfig.withMarket(MarketType.ON_DEMAND);
            }
            if (slaveCount > 0) {
                this.sparkJobFlowInstancesConfig = this.sparkJobFlowInstancesConfig.withInstanceGroups(Arrays.asList(masterConfig, slaveConfig));
            } else {
                this.sparkJobFlowInstancesConfig = this.sparkJobFlowInstancesConfig.withInstanceGroups(slaveConfig);
            }

            this.sparkRunJobFlowRequest = new RunJobFlowRequest();
            if (!sparkEmrConfigs.isEmpty()) {
                List<Configuration> emrConfigs = new ArrayList<>();
                for (EmrConfig config : sparkEmrConfigs) {
                    emrConfigs.add(config.toAwsConfig());
                }
                this.sparkRunJobFlowRequest = this.sparkRunJobFlowRequest.withConfigurations(emrConfigs);
            }
            this.sparkRunJobFlowRequest =
                    this.sparkRunJobFlowRequest.withName(sparkClusterName).withApplications((new Application()).withName("Spark"))
                            .withReleaseLabel(sparkEmrRelease)
                            .withServiceRole(sparkEmrServiceRole)
                            .withJobFlowRole(sparkInstanceRole)
                            .withInstances(this.sparkJobFlowInstancesConfig);

            return new SparkEMRClient(
                    this.sparkClusterName,
                    this.sparkAwsRegion,
                    this.sparkEmrRelease,
                    this.sparkEmrServiceRole,
                    this.sparkEmrConfigs,
                    this.sparkSubNetid,
                    this.sparkSecurityGroupIds,
                    this.sparkInstanceCount,
                    this.sparkInstanceType,
                    this.sparkInstanceBidPrice,
                    this.sparkInstanceRole,
                    this.sparkS3JarFolder,
                    this.sparkTimeoutDurationMinutes,
                    this.sparkEmrClientBuilder,
                    this.sparkS3ClientBuilder,
                    this.sparkJobFlowInstancesConfig,
                    this.sparkRunJobFlowRequest,
                    this.sparkS3PutObjectDecorator,
                    this.sparkSubmitConfs
            );
        }

    }

}
