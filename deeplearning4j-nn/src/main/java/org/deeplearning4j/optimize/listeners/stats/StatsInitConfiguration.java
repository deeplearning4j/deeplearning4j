package org.deeplearning4j.optimize.listeners.stats;

/**
 * Created by Alex on 29/09/2016.
 */
public interface StatsInitConfiguration {

    //OS, JVM, ND4J backend
    boolean collectMachineInfo();

    //Configuration, number of parameters, etc.
    boolean collectModelInfo();

}
