/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  * *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.nativeimage;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ops.NoOp;
import org.nd4j.linalg.factory.InitializationController;
import org.nd4j.linalg.factory.Nd4jInitialization;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Nd4jInitializationLifecycleTest {

    private static final String NATIVE_IMAGE_PROPERTIES =
            "META-INF/native-image/org.eclipse.deeplearning4j/"
                    + "nd4j-api/native-image.properties";

    @Test
    public void operationPrototypeConstructionDoesNotBootstrapNd4j() {
        assertEquals(InitializationController.Phase.NEW,
                Nd4jInitialization.getPhase());

        assertNotNull(new NoOp());

        assertEquals(InitializationController.Phase.NEW,
                Nd4jInitialization.getPhase(),
                "Constructing an operation prototype must not initialize ND4J");
    }

    @Test
    public void initializationControllerRejectsReentryAndRetainsFirstFailure() {
        InitializationController controller =
                new InitializationController("test component");

        assertTrue(controller.begin());
        assertEquals(InitializationController.Phase.INITIALIZING,
                controller.getPhase());
        assertTrue(controller.isInitializingByCurrentThread());
        assertThrows(IllegalStateException.class, controller::begin);

        IllegalArgumentException firstFailure =
                new IllegalArgumentException("first failure");
        controller.fail(firstFailure);

        assertEquals(InitializationController.Phase.FAILED,
                controller.getPhase());
        assertSame(firstFailure, controller.getFailure());
        assertSame(firstFailure,
                assertThrows(IllegalArgumentException.class,
                        controller::throwIfFailed));
        assertSame(firstFailure,
                assertThrows(IllegalArgumentException.class,
                        controller::begin));
    }

    @Test
    public void concurrentCallerObservesOnlyCommittedReadiness() throws Exception {
        InitializationController controller =
                new InitializationController("concurrent component");
        assertTrue(controller.begin());

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> secondInitializer = executor.submit(controller::begin);
            controller.complete();

            assertFalse(secondInitializer.get(),
                    "A concurrent caller must observe the committed READY state");
            assertEquals(InitializationController.Phase.READY,
                    controller.getPhase());
            assertTrue(controller.isReady());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    public void nativeImageDefersLifecycleStateToRuntime() throws IOException {
        Properties properties = new Properties();
        try (InputStream input = getClass().getClassLoader()
                .getResourceAsStream(NATIVE_IMAGE_PROPERTIES)) {
            assertNotNull(input, "Missing ND4J native-image properties");
            properties.load(input);
        }

        String args = properties.getProperty("Args");
        assertNotNull(args);
        assertTrue(args.contains(
                "--initialize-at-run-time=org.nd4j.linalg.factory.Nd4jInitialization,"
                        + "org.nd4j.linalg.factory.InitializationController"));
    }
}
