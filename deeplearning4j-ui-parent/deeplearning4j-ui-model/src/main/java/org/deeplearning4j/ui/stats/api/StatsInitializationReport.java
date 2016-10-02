package org.deeplearning4j.ui.stats.api;

import org.deeplearning4j.ui.stats.StatsListener;

/**
 * An interface used with {@link StatsListener} for reporting static information.
 * The idea is that this information will be reported only once, at the first call of the StatsListener. Comparatively,
 * the {@link StatsReport} will be used multiple times - every N iterations according to the configuration ({@link StatsListenerConfiguration}).
 * <p>
 * Note that the software, hardware and model information may or may not be obtained and reported, depending on the configuration
 * provided by the relevant {@link StatsInitializationConfiguration}
 *
 * @author Alex Black
 */
public interface StatsInitializationReport {

    /**
     * @param arch             Operating system architecture, as reported by JVM
     * @param osName           Operating system name
     * @param jvmName          JVM name
     * @param jvmVersion       JVM version
     * @param jvmSpecVersion   JVM Specification version (for example, 1.8)
     * @param nd4jBackendClass ND4J backend Factory class
     * @param nd4jDataTypeName ND4J datatype name
     */
    void reportSoftwareInfo(String arch, String osName, String jvmName, String jvmVersion, String jvmSpecVersion,
                            String nd4jBackendClass, String nd4jDataTypeName);

    /**
     * @param jvmAvailableProcessors Number of available processor cores according to the JVM
     * @param numDevices             Number of compute devices (GPUs)
     * @param deviceTotalMemory      GPU memory by device: same length as numDevices. May be null, if numDevices is 0
     */
    void reportHardwareInfo(int jvmAvailableProcessors, int numDevices, long jvmMaxMemory, long offHeapMaxMemory,
                            long[] deviceTotalMemory, String[] deviceDescription);


    /**
     * Report the model information
     *
     * @param modelClassName  Model class name: i.e., type of model
     * @param modelConfigJson Model configuration, as JSON string
     * @param numLayers       Number of layers in the model
     * @param numParams       Number of parameters in the model
     */
    void reportModelInfo(String modelClassName, String modelConfigJson, String[] paramNames, int numLayers, long numParams);

    /**
     * Convert the initialization report to a byte[] for storage
     */
    byte[] toByteArray();

    /**
     * Deserialize the Stats from a byte[]
     *
     * @param byteArray Source array to deserialize from
     */
    void fromByteArray(byte[] byteArray);


    boolean hasSoftwareInfo();

    boolean hasHardwareInfo();

    boolean hasModelInfo();

    String getSwArch();

    String getSwOsName();

    String getSwJvmName();

    String getSwJvmVersion();

    String getSwJvmSpecVersion();

    String getSwNd4jBackendClass();

    String getSwNd4jDataTypeName();

    int getHwJvmAvailableProcessors();

    int getHwNumDevices();

    long getHwJvmMaxMemory();

    long getHwOffHeapMaxMemory();

    long[] getHwDeviceTotalMemory();

    String[] getHwDeviceDescription();

    String getModelClassName();

    String getModelConfigJson();

    String[] getModelParamNames();

    int getModelNumLayers();

    long getModelNumParams();

}
