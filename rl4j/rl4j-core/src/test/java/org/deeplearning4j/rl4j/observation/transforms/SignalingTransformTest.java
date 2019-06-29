package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class SignalingTransformTest {

    @Test(expected = NullPointerException.class)
    public void when_builderListenerIsNull_expect_NullPointerException() {
        SignalingTransform sut = SignalingTransform.builder()
                .listener(null)
                .build();
    }

    @Test(expected = NullPointerException.class)
    public void when_methodListenerIsNull_expect_NullPointerException() {
        SignalingTransform sut = SignalingTransform.builder().build();
        sut.addListener(null);

    }


    @Test
    public void when_resetIsCalled_expect_onResetIsCalled() {
        // Arrange
        TestTransformListener testListenerFromBuilder = new TestTransformListener();
        TestTransformListener testListenerFromMethod = new TestTransformListener();
        SignalingTransform sut = SignalingTransform.builder()
            .listener(testListenerFromBuilder)
            .build();
        sut.addListener(testListenerFromMethod);

        // Act
        sut.reset();

        // Assert
        assertTrue(testListenerFromBuilder.isOnResetCalled);
        assertTrue(testListenerFromMethod.isOnResetCalled);
        assertFalse(testListenerFromBuilder.isOnTransformCalled);
        assertFalse(testListenerFromMethod.isOnTransformCalled);
    }

    @Test
    public void when_transformIsCalled_expect_onTransformIsCalled() {
        // Arrange
        TestTransformListener testListenerFromBuilder = new TestTransformListener();
        TestTransformListener testListenerFromMethod = new TestTransformListener();
        SignalingTransform sut = SignalingTransform.builder()
                .listener(testListenerFromBuilder)
                .build();
        sut.addListener(testListenerFromMethod);

        // Act
        sut.transform(new SimpleObservation(Nd4j.create(new double[] { 123.0 })));

        // Assert
        assertFalse(testListenerFromBuilder.isOnResetCalled);
        assertFalse(testListenerFromMethod.isOnResetCalled);
        assertTrue(testListenerFromBuilder.isOnTransformCalled);
        assertTrue(testListenerFromMethod.isOnTransformCalled);
    }

    private static class TestTransformListener implements TransformListener {

        public boolean isOnResetCalled;
        public boolean isOnTransformCalled;

        @Override
        public void onReset() {
            isOnResetCalled = true;
        }

        @Override
        public void onTransform(Observation observation) {
            isOnTransformCalled = true;
        }
    }
}
