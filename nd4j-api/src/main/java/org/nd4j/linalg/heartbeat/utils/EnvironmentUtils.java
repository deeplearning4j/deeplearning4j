package org.nd4j.linalg.heartbeat.utils;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Environment;

/**
 * @author raver119@gmail.com
 */
public class EnvironmentUtils {

    /**
     * This method build
     * @return
     */
    public static Environment buildEnvironment() {
        Environment environment = new Environment();

        environment.setJavaVersion(System.getProperty("java.specification.version"));
        environment.setNumCores(Runtime.getRuntime().availableProcessors());
        environment.setAvailableMemory(Runtime.getRuntime().maxMemory());
        environment.setOsArch(System.getProperty("os.arch"));
        environment.setOsName(System.getProperty("os.name"));
        environment.setBackendUsed(Nd4j.getExecutioner().getClass().getSimpleName());

        return environment;
    }
}
