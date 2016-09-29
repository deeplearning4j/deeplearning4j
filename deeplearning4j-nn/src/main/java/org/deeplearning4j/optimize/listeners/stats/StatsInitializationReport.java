package org.deeplearning4j.optimize.listeners.stats;

/**
 * Created by Alex on 29/09/2016.
 */
public interface StatsInitializationReport {

    void reportMachineInfo(int availableProcessors, String arch, String osName, String jvmName, String jvmVersion, String jvmSpecVersion,
                           String nd4jBackendClass, String nd4jDataTypeName);

    void reportModelInfo(String networkConfigJson, int numLayers, int numParams);

}
