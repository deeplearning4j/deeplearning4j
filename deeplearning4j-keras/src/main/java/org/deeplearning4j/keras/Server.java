package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

/**
 * Main class for the DL4J-as-Keras-backend. Simply launches py4j GatewayServer with
 * an entry point exposing API available on Python side.
 *
 * @author pkoperek@gmail.com
 */
@Slf4j
public class Server {

    public static void main(String[] args) {

        try {
            Nd4j.create(1); // ensures ND4J is on classpath
        } catch (Throwable e) {
            log.error("Failed to start DL4J gateway.", e);
            System.exit(1);
        }

        GatewayServer gatewayServer = new GatewayServer(new DeepLearning4jEntryPoint());
        gatewayServer.start();
        log.info("DeepLearning4j Gateway server running");
    }
}
