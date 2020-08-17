package org.deeplearning4j.rl4j.builder;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class BaseAgentLearnerBuilderTest {
    @Mock
    BaseAgentLearnerBuilder.Configuration configuration;

    @Mock
    ITrainableNeuralNet neuralNet;

    @Mock
    Builder<Environment<Integer>> environmentBuilder;

    @Mock
    Builder<TransformProcess> transformProcessBuilder;

    @Mock
    IUpdateAlgorithm updateAlgorithmMock;

    @Mock
    INeuralNetUpdater neuralNetUpdaterMock;

    @Mock
    ExperienceHandler experienceHandlerMock;

    @Mock
    Environment environmentMock;

    @Mock
    IPolicy policyMock;

    @Mock
    TransformProcess transformProcessMock;

    BaseAgentLearnerBuilder sut;

    @Before
    public void setup() {
        sut = mock(
                BaseAgentLearnerBuilder.class,
                Mockito.withSettings()
                        .useConstructor(configuration, neuralNet, environmentBuilder, transformProcessBuilder)
                        .defaultAnswer(Mockito.CALLS_REAL_METHODS)
        );

        AgentLearner.Configuration agentLearnerConfiguration = AgentLearner.Configuration.builder().maxEpisodeSteps(200).build();

        when(sut.buildUpdateAlgorithm()).thenReturn(updateAlgorithmMock);
        when(sut.buildNeuralNetUpdater()).thenReturn(neuralNetUpdaterMock);
        when(sut.buildExperienceHandler()).thenReturn(experienceHandlerMock);
        when(environmentBuilder.build()).thenReturn(environmentMock);
        when(transformProcessBuilder.build()).thenReturn(transformProcessMock);
        when(sut.buildPolicy()).thenReturn(policyMock);
        when(configuration.getAgentLearnerConfiguration()).thenReturn(agentLearnerConfiguration);
    }

    @Test
    public void when_buildingAgentLearner_expect_dependenciesAndAgentLearnerIsBuilt() {
        // Arrange

        // Act
        sut.build();

        // Assert
        verify(environmentBuilder, times(1)).build();
        verify(transformProcessBuilder, times(1)).build();
        verify(sut, times(1)).buildPolicy();
        verify(sut, times(1)).buildExperienceHandler();
        verify(sut, times(1)).buildUpdateAlgorithm();
        verify(sut, times(1)).buildNeuralNetUpdater();
        verify(sut, times(1)).buildAgentLearner();
    }

}
