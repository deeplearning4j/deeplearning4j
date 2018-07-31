package org.deeplearning4j.config;

public class DL4JEnvironmentVars {

    private DL4JEnvironmentVars(){ }


    /**
     * Applicability: Module dl4j-spark-parameterserver_2.xx<br>
     * Usage: A fallback for determining the local IP for a Spark training worker, if other approaches
     * fail to determine the local IP
     */
    public static final String DL4J_VOID_IP = "DL4J_VOID_IP";

}
