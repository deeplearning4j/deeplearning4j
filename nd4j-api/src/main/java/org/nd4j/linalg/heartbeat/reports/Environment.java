package org.nd4j.linalg.heartbeat.reports;

import lombok.Data;

import java.io.Serializable;

/**
 * Bean/POJO that describes current jvm/node
 *
 * @author raver119@gmail.com
 */
@Data
public class Environment implements Serializable {
    private long serialVersionID;

    /*
        number of cores available to jvm
     */
    private int numCores;

    /*
        memory available within current process
     */
    private long availableMemory;

    /*
        System.getPriority("java.specification.version");
     */
    private String javaVersion;


    /*
        System.getProperty("os.name");
     */
    private String osName;

    /*
        System.getProperty("os.arch");
        however, in 97% of cases it will be amd64. it will be better to have JNI call for that in future
     */
    private String osArch;

    /*
        Nd4j backend being used within current JVM session
    */
    private String backendUsed;
}
