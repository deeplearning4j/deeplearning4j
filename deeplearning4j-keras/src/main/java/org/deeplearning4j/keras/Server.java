package org.deeplearning4j.keras;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

public class Server {

    private static final Logger logger = LoggerFactory.getLogger(Server.class);

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new DeepLearning4jEntryPoint());
        gatewayServer.start();
        logger.info("DeepLearning4j Gateway server running");
    }
}
