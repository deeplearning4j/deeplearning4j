/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.agent.Agent;
import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic.AdvantageActorCritic;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.builder.AdvantageActorCriticBuilder;
import org.deeplearning4j.rl4j.builder.DoubleDQNBuilder;
import org.deeplearning4j.rl4j.builder.NStepQLearningBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.mdp.robotlake.RobotLake;
import org.deeplearning4j.rl4j.network.ActorCriticNetwork;
import org.deeplearning4j.rl4j.network.ChannelToNetworkInputMapper;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.QNetwork;
import org.deeplearning4j.rl4j.network.ac.*;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.trainer.ITrainer;
import org.deeplearning4j.rl4j.trainer.SyncTrainer;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class RobotLakeExample {
    private static final int FROZEN_LAKE_SIZE = 5;

    private static final int NUM_EPISODES = 10000;

    private static final ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] INPUT_CHANNEL_BINDINGS = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("tracker-in", "tracker"),
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("radar-in", "radar"),
    };
    private static final String[] TRANSFORM_PROCESS_OUTPUT_CHANNELS = new String[] { "tracker", "radar" };

    public static void main(String[] args) {
        Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);
        Builder<Environment<Integer>> environmentBuilder = () -> new RobotLake(FROZEN_LAKE_SIZE, false, rnd);
        Builder<TransformProcess> transformProcessBuilder = () -> TransformProcess.builder()
                .transform("tracker", new ArrayToINDArrayTransform())
                .transform("radar", new ArrayToINDArrayTransform())
                .build(TRANSFORM_PROCESS_OUTPUT_CHANNELS);

        List<AgentListener<Integer>> listeners = new ArrayList<AgentListener<Integer>>() {
            {
                add(new EpisodeScorePrinter(100));
            }
        };

        Builder<IAgentLearner<Integer>> builder = setupDoubleDQN(environmentBuilder, transformProcessBuilder, listeners, rnd);
        //Builder<IAgentLearner<Integer>> builder = setupNStepQLearning(environmentBuilder, transformProcessBuilder, listeners, rnd);
        //Builder<IAgentLearner<Integer>> builder = setupAdvantageActorCritic(environmentBuilder, transformProcessBuilder, listeners, rnd);

        ITrainer trainer = SyncTrainer.<Integer>builder()
                .agentLearnerBuilder(builder)
                .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                .build();

        long before = System.nanoTime();
        trainer.train();
        long after = System.nanoTime();


        System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
    }

    private static Builder<IAgentLearner<Integer>> setupDoubleDQN(Builder<Environment<Integer>> environmentBuilder,
                                                                  Builder<TransformProcess> transformProcessBuilder,
                                                                  List<AgentListener<Integer>> listeners,
                                                                  Random rnd) {
        ITrainableNeuralNet network = buildQNetwork();

        DoubleDQNBuilder.Configuration configuration = DoubleDQNBuilder.Configuration.builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(3000)
                        .minEpsilon(0.1)
                        .build())
                .experienceHandlerConfiguration(ReplayMemoryExperienceHandler.Configuration.builder()
                        .maxReplayMemorySize(10000)
                        .batchSize(64)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(50)
                        .build())
                .updateAlgorithmConfiguration(BaseTransitionTDAlgorithm.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .build();
        return new DoubleDQNBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }


    private static Builder<IAgentLearner<Integer>> setupNStepQLearning(Builder<Environment<Integer>> environmentBuilder,
                                                                       Builder<TransformProcess> transformProcessBuilder,
                                                                       List<AgentListener<Integer>> listeners,
                                                                       Random rnd) {
        ITrainableNeuralNet network = buildQNetwork();

        NStepQLearningBuilder.Configuration configuration = NStepQLearningBuilder.Configuration.builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(75000)
                        .minEpsilon(0.1)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(50)
                        .build())
                .nstepQLearningConfiguration(NStepQLearning.Configuration.builder()
                        .build())
                .experienceHandlerConfiguration(StateActionExperienceHandler.Configuration.builder()
                        .batchSize(Integer.MAX_VALUE)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(100)
                        .build())
                .agentLearnerListeners(listeners)
                .build();
        return new NStepQLearningBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }

    private static Builder<IAgentLearner<Integer>> setupAdvantageActorCritic(Builder<Environment<Integer>> environmentBuilder,
                                                                             Builder<TransformProcess> transformProcessBuilder,
                                                                             List<AgentListener<Integer>> listeners,
                                                                             Random rnd) {
        ITrainableNeuralNet network = buildActorCriticNetwork();

        AdvantageActorCriticBuilder.Configuration configuration = AdvantageActorCriticBuilder.Configuration.builder()
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .build())
                .advantageActorCriticConfiguration(AdvantageActorCritic.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .experienceHandlerConfiguration(StateActionExperienceHandler.Configuration.builder()
                        .batchSize(Integer.MAX_VALUE)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(100)
                        .build())
                .agentLearnerListeners(listeners)
                .build();
        return new AdvantageActorCriticBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }

    private static ComputationGraphConfiguration.GraphBuilder buildBaseNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .setInputTypes(InputType.feedForward(2), // tracker
                               InputType.feedForward(4)) // radar )
                .addInputs("tracker-in", "radar-in")

                .layer("dl_1", new DenseLayer.Builder().activation(Activation.RELU).nOut(40).build(), "tracker-in", "radar-in")
                .layer("dl_out", new DenseLayer.Builder().activation(Activation.RELU).nOut(40).build(), "dl_1");
    }

    private static ITrainableNeuralNet buildQNetwork() {
        ComputationGraphConfiguration conf = buildBaseNetworkConfiguration()
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                        .nOut(RobotLake.NUM_ACTIONS).build(), "dl_out")

                .setOutputs("output")
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        return QNetwork.builder()
                .withNetwork(model)
                .inputBindings(INPUT_CHANNEL_BINDINGS)
                .channelNames(TRANSFORM_PROCESS_OUTPUT_CHANNELS)
                .build();
    }

    private static ITrainableNeuralNet buildActorCriticNetwork() {
        ComputationGraphConfiguration conf = buildBaseNetworkConfiguration()
                .addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                        .nOut(1).build(), "dl_out")
                .addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                        .nOut(RobotLake.NUM_ACTIONS).build(), "dl_out")
                .setOutputs("value", "softmax")
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return ActorCriticNetwork.builder()
                .withCombinedNetwork(model)
                .inputBindings(INPUT_CHANNEL_BINDINGS)
                .channelNames(TRANSFORM_PROCESS_OUTPUT_CHANNELS)
                .build();
    }

    private static class EpisodeScorePrinter implements AgentListener<Integer> {
        private final int trailingNum;
        private final boolean[] results;
        private int episodeCount;

        public EpisodeScorePrinter(int trailingNum) {
            this.trailingNum = trailingNum;
            results = new boolean[trailingNum];
        }
        
        @Override
        public ListenerResponse onBeforeEpisode(Agent agent) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onBeforeStep(Agent agent, Observation observation, Integer integer) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onAfterStep(Agent agent, StepResult stepResult) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public void onAfterEpisode(Agent agent) {
            RobotLake environment = (RobotLake)agent.getEnvironment();
            boolean isSuccess = false;

            String result;
            if(environment.isGoalReached()) {
                result = "GOAL REACHED ******";
                isSuccess = true;
            } else if(environment.isEpisodeFinished()) {
                result = "FAILED";
            } else {
                result = "DID NOT FINISH";
            }

            results[episodeCount % trailingNum] = isSuccess;

            if(episodeCount >= trailingNum) {
                int successCount = 0;
                for (int i = 0; i < trailingNum; ++i) {
                    successCount += results[i] ? 1 : 0;
                }
                double successRatio = successCount / (double)trailingNum;

                System.out.println(String.format("[%s] Episode %4d : score = %4.2f, success ratio = %4.2f, result = %s", agent.getId(), episodeCount, agent.getReward(), successRatio, result));
            } else {
                System.out.println(String.format("[%s] Episode %4d : score = %4.2f, result = %s", agent.getId(), episodeCount, agent.getReward(), result));
            }


            ++episodeCount;
        }
    }
}