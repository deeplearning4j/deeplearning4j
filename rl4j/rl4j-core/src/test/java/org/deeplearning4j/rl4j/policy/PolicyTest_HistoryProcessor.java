package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.learning.TestHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.deeplearning4j.rl4j.learning.sync.qlearning.TestQLearningDiscrete;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class PolicyTest_HistoryProcessor {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Test
    public void play_WithHistoryProcessor_ShouldCallRecord() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestPolicy sut = new TestPolicy();

        // Act
        sut.play(mdp, hp);

        assertEquals(36, hp.recordCallCount);
    }

    @Test
    public void play_WithHistoryProcessor_ShouldCallAdd() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestPolicy sut = new TestPolicy();

        // Act
        sut.play(mdp, hp);

        assertEquals(19, hp.addCallCount);
    }

    @Test
    public void play_WithHistoryProcessor_ShouldCallGetHistory() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestPolicy sut = new TestPolicy();

        // Act
        sut.play(mdp, hp);

        assertEquals(10, hp.getHistoryCallCount);
    }

    @Test
    public void play_NoHistoryProcessor_RewardShouldBe() throws IOException {

        // Arrange
        TestHistoryProcessor hp = null;
        TestMDP mdp = new TestMDP();
        TestPolicy sut = new TestPolicy();

        // Act
        double reward = sut.play(mdp, hp);

        assertEquals(9.0, reward, 0.1);
    }

    @Test
    public void play_WithHistoryProcessor_RewardShouldBe() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestPolicy sut = new TestPolicy();

        // Act
        double reward = sut.play(mdp, hp);

        assertEquals(36.0, reward, 0.1);
    }

}
