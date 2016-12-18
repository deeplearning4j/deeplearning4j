package org.deeplearning4j;

import py4j.GatewayServer;

public class Server {
    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new DeepLearning4jEntryPoint());
        gatewayServer.start();
        System.out.println("DeepLearning4j Gateway server running");
    }
}
