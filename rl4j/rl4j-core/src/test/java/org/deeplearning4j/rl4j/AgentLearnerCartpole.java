/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
import org.deeplearning4j.rl4j.mdp.CartpoleEnvironment;
import org.deeplearning4j.rl4j.network.ActorCriticNetwork;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.QNetwork;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.network.ac.ActorCriticSeparate;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.trainer.AsyncTrainer;
import org.deeplearning4j.rl4j.trainer.ITrainer;
import org.deeplearning4j.rl4j.trainer.SyncTrainer;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.ArrayList;
import java.util.List;

public class AgentLearnerCartpole {

    private static final boolean IS_ASYNC = false;
    private static final int NUM_THREADS = 2;
    private static final boolean USE_SEPARATE_NETWORKS = false;

    private static final int NUM_EPISODES = 3000;

    public static void main(String[] args) {

        Builder<Environment<Integer>> environmentBuilder = CartpoleEnvironment::new;
        Builder<TransformProcess> transformProcessBuilder = () -> TransformProcess.builder()
                .transform("data", new ArrayToINDArrayTransform())
                .build("data");

        Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);

        List<AgentListener<Integer>> listeners = new ArrayList<AgentListener<Integer>>() {
            {
                add(new EpisodeScorePrinter());
            }
        };

        //Builder<IAgentLearner<Integer>> builder = setupDoubleDQN(environmentBuilder, transformProcessBuilder, listeners, rnd);
        Builder<IAgentLearner<Integer>> builder = setupNStepQLearning(environmentBuilder, transformProcessBuilder, listeners, rnd);
        //Builder<IAgentLearner<Integer>> builder = setupAdvantageActorCritic(environmentBuilder, transformProcessBuilder, listeners, rnd);

        ITrainer trainer;
        if(IS_ASYNC) {
            trainer = AsyncTrainer.<Integer>builder()
                    .agentLearnerBuilder(builder)
                    .numThreads(NUM_THREADS)
                    .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                    .build();
        } else {
            trainer = SyncTrainer.<Integer>builder()
                    .agentLearnerBuilder(builder)
                    .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                    .build();
        }

        long before = System.nanoTime();
        trainer.train();
        long after = System.nanoTime();

        System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
    }

    private static Builder<IAgentLearner<Integer>> setupDoubleDQN(Builder<Environment<Integer>> environmentBuilder,
                                                                  Builder<TransformProcess> transformProcessBuilder,
                                                                  List<AgentListener<Integer>> listeners,
                                                                  Random rnd) {
        ITrainableNeuralNet network = buildDQNNetwork();

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
                .asynchronous(IS_ASYNC)
                .build();
        return new DoubleDQNBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }

    private static Builder<IAgentLearner<Integer>> setupNStepQLearning(Builder<Environment<Integer>> environmentBuilder,
                                                                       Builder<TransformProcess> transformProcessBuilder,
                                                                       List<AgentListener<Integer>> listeners,
                                                                       Random rnd) {
        ITrainableNeuralNet network = buildDQNNetwork();

        NStepQLearningBuilder.Configuration configuration = NStepQLearningBuilder.Configuration.builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(75000  / (IS_ASYNC ? NUM_THREADS : 1))
                        .minEpsilon(0.1)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(50)
                        .build())
                .nstepQLearningConfiguration(NStepQLearning.Configuration.builder()
                        .build())
                .experienceHandlerConfiguration(StateActionExperienceHandler.Configuration.builder()
                        .batchSize(5)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .asynchronous(IS_ASYNC)
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
                        .batchSize(5)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .asynchronous(IS_ASYNC)
                .build();
        return new AdvantageActorCriticBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }

    private static ITrainableNeuralNet buildDQNNetwork() {
        DQNDenseNetworkConfiguration netConf = DQNDenseNetworkConfiguration.builder()
                .updater(new Adam())
                .numHiddenNodes(40)
                .numLayers(2)
                .build();
        DQNFactory factory = new DQNFactoryStdDense(netConf);
        IDQN dqnNetwork = factory.buildDQN(new int[] { 4 }, 2);
        return QNetwork.builder().withNetwork((MultiLayerNetwork)dqnNetwork.getNeuralNetworks()[0]).build();
    }

    private static ITrainableNeuralNet buildActorCriticNetwork() {
        ActorCriticDenseNetworkConfiguration netConf =  ActorCriticDenseNetworkConfiguration.builder()
                .updater(new Adam())
                .numHiddenNodes(40)
                .numLayers(2)
                .build();

        if(USE_SEPARATE_NETWORKS) {
            ActorCriticFactorySeparateStdDense factory = new ActorCriticFactorySeparateStdDense(netConf);
            ActorCriticSeparate network =  factory.buildActorCritic(new int[] { 4 }, 2);
            return ActorCriticNetwork.builder()
                    .withSeparateNetworks((MultiLayerNetwork)network.getNeuralNetworks()[0], (MultiLayerNetwork)network.getNeuralNetworks()[1])
                    .build();
        }

        ActorCriticFactoryCompGraphStdDense factory = new ActorCriticFactoryCompGraphStdDense(netConf);
        ActorCriticCompGraph network = factory.buildActorCritic(new int[] { 4 }, 2);
        return ActorCriticNetwork.builder()
                    .withCombinedNetwork((ComputationGraph) network.getNeuralNetworks()[0])
                    .build();
    }

    private static class EpisodeScorePrinter implements AgentListener<Integer> {
        private int episodeCount;
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
            System.out.println(String.format("[%s] Episode %4d : score = %3d", agent.getId(), episodeCount, (int)agent.getReward()));
            ++episodeCount;
        }
    }
}