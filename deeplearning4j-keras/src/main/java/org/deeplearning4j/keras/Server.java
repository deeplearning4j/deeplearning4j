package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
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
        GatewayServer gatewayServer = new GatewayServer(new DeepLearning4jEntryPoint());
        gatewayServer.start();
        log.info("DeepLearning4j Gateway server running");
    }
}
