package org.deeplearning4j.rl4j;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.agent.Agent;
import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic.AdvantageActorCritic;
import org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.builder.AdvantageActorCriticBuilder;
import org.deeplearning4j.rl4j.builder.NStepQLearningBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.mdp.TMazeEnvironment;
import org.deeplearning4j.rl4j.network.ActorCriticNetwork;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.QNetwork;
import org.deeplearning4j.rl4j.network.ac.ActorCriticLoss;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.trainer.AsyncTrainer;
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
import java.util.concurrent.atomic.AtomicInteger;

public class TMazeExample {

    private static final boolean IS_ASYNC = false;
    private static final int NUM_THREADS = 2;

    private static final int TMAZE_LENGTH = 10;

    private static final int NUM_INPUTS = 5;
    private static final int NUM_ACTIONS = 4;

    private static final double MIN_EPSILON = 0.1;

    private static final int NUM_EPISODES = 3000;

    public static void main(String[] args) {

        Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);

        Builder<Environment<Integer>> environmentBuilder = () -> new TMazeEnvironment(TMAZE_LENGTH, rnd);
        Builder<TransformProcess> transformProcessBuilder = () -> TransformProcess.builder()
                .transform("data", new ArrayToINDArrayTransform(1, NUM_INPUTS, 1))
                .build("data");

        List<AgentListener<Integer>> listeners = new ArrayList<AgentListener<Integer>>() {
            {
                add(new EpisodeScorePrinter(25)); // compute the success rate with the trailing 25 episodes.
            }
        };

//        Builder<IAgentLearner<Integer>> builder = setupNStepQLearning(environmentBuilder, transformProcessBuilder, listeners, rnd);
        Builder<IAgentLearner<Integer>> builder = setupAdvantageActorCritic(environmentBuilder, transformProcessBuilder, listeners, rnd);

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

    private static Builder<IAgentLearner<Integer>> setupNStepQLearning(Builder<Environment<Integer>> environmentBuilder,
                                                                       Builder<TransformProcess> transformProcessBuilder,
                                                                       List<AgentListener<Integer>> listeners,
                                                                       Random rnd) {
        ITrainableNeuralNet network = buildQNetwork();

        NStepQLearningBuilder.Configuration configuration = NStepQLearningBuilder.Configuration.builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(25000 / (IS_ASYNC ? NUM_THREADS : 1))
                        .minEpsilon(MIN_EPSILON)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(25)
                        .build())
                .nstepQLearningConfiguration(NStepQLearning.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .experienceHandlerConfiguration(StateActionExperienceHandler.Configuration.builder()
                        .batchSize(Integer.MAX_VALUE)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(40)
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
                        .batchSize(Integer.MAX_VALUE)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .maxEpisodeSteps(40)
                        .build())
                .agentLearnerListeners(listeners)
                .asynchronous(IS_ASYNC)
                .build();
        return new AdvantageActorCriticBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }

    private static ComputationGraphConfiguration.GraphBuilder buildBaseNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Adam())
                        .weightInit(WeightInit.XAVIER)
                        .graphBuilder()
                        .setInputTypes(InputType.recurrent(NUM_INPUTS))
                        .addInputs("input")
                        .addLayer("goal", new LSTM.Builder()
                                .nOut(40)
                                .activation(Activation.TANH)
                                .build(), "input")
                        .addLayer("corridor", new DenseLayer.Builder().nOut(40).activation(Activation.RELU).build(), "input", "goal")
                        .addLayer("corridor-1", new DenseLayer.Builder().nOut(20).activation(Activation.RELU).build(), "corridor")
                        .addVertex("corridor-rnn", new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), "corridor-1");
    }

    private static ITrainableNeuralNet buildQNetwork() {
        ComputationGraphConfiguration conf = buildBaseNetworkConfiguration()
                        .addLayer("output", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                                .nOut(NUM_ACTIONS).build(), "goal", "corridor-rnn")

                        .setOutputs("output")
                        .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        return QNetwork.builder()
                .withNetwork(model)
                .build();
    }

    private static ITrainableNeuralNet buildActorCriticNetwork() {
        ComputationGraphConfiguration conf = buildBaseNetworkConfiguration()
                        .addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                                .nOut(1).build(), "goal", "corridor-rnn")
                        .addLayer("softmax", new RnnOutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                                .nOut(NUM_ACTIONS).build(), "goal", "corridor-rnn")
                        .setOutputs("value", "softmax")
                        .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return ActorCriticNetwork.builder()
                .withCombinedNetwork(model)
                .build();
    }

    private static class EpisodeScorePrinter implements AgentListener<Integer> {
        private final boolean[] results;
        private final AtomicInteger episodeCount = new AtomicInteger(0);
        private final int trailingNum;

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
            TMazeEnvironment environment = (TMazeEnvironment)agent.getEnvironment();
            int currentEpisodeCount = episodeCount.getAndIncrement();
            results[currentEpisodeCount % trailingNum] = environment.hasNavigatedToSolution();

            String stateAtEnd;
            if(environment.hasNavigatedToSolution()) {
                stateAtEnd = "Reached GOAL";
            } else if(environment.isEpisodeFinished()) {
                stateAtEnd = "Reached TRAP";
            } else {
                stateAtEnd = "Did not finish";
            }

            if(currentEpisodeCount >= trailingNum) {
                int successCount = 0;
                for (int i = 0; i < trailingNum; ++i) {
                    successCount += results[i] ? 1 : 0;
                }
                double successRatio = successCount / (double)trailingNum;
                System.out.println(String.format("[%s] Episode %4d : score = %6.2f success ratio = %4.2f %s", agent.getId(), currentEpisodeCount, agent.getReward(), successRatio, stateAtEnd ));
            } else {
                System.out.println(String.format("[%s] Episode %4d : score = %6.2f %s", agent.getId(), currentEpisodeCount, agent.getReward(), stateAtEnd ));
            }
        }
    }
}