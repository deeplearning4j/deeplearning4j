package org.deeplearning4j.ui.stats.api;

import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.ui.stats.StatsListener;

import java.util.Map;

/**
 * An interface used with {@link StatsListener} for reporting static information.
 * The idea is that this information will be reported only once, at the first call of the StatsListener. Comparatively,
 * the {@link StatsReport} will be used multiple times - every N iterations according to the configuration ({@link StatsUpdateConfiguration}).
 * <p>
 * Note that the software, hardware and model information may or may not be obtained and reported, depending on the configuration
 * provided by the relevant {@link StatsInitializationConfiguration}
 *
 * @author Alex Black
 */
public interface StatsInitializationReport extends Persistable {

    void reportIDs(String sessionID, String typeID, String workerID, long timestamp);

    /**
     * @param arch             Operating system architecture, as reported by JVM
     * @param osName           Operating system name
     * @param jvmName          JVM name
     * @param jvmVersion       JVM version
     * @param jvmSpecVersion   JVM Specification version (for example, 1.8)
     * @param nd4jBackendClass ND4J backend Factory class
     * @param nd4jDataTypeName ND4J datatype name
     * @param hostname         Hostname for the machine, if available
     * @param jvmUID           A unique identified for the current JVM. Should be shared by all instances in the same JVM.
     *                         Should vary for different JVMs on the same machine.
     * @param swEnvironmentInfo Environment information: Usually from Nd4j.getExecutioner().getEnvironmentInformation()
     */
    void reportSoftwareInfo(String arch, String osName, String jvmName, String jvmVersion, String jvmSpecVersion,
                    String nd4jBackendClass, String nd4jDataTypeName, String hostname, String jvmUID,
                    Map<String, String> swEnvironmentInfo);

    /**
     * @param jvmAvailableProcessors Number of available processor cores according to the JVM
     * @param numDevices             Number of compute devices (GPUs)
     * @param jvmMaxMemory           Maximum memory for the JVM
     * @param offHeapMaxMemory       Maximum off-heap memory
     * @param deviceTotalMemory      GPU memory by device: same length as numDevices. May be null, if numDevices is 0
     * @param deviceDescription      Description of each device. May be null, if numDevices is 0
     * @param hardwareUID            A unique identifier for the machine. Should be shared by all instances running on
     *                               the same machine, including in different JVMs
     *
     */
    void reportHardwareInfo(int jvmAvailableProcessors, int numDevices, long jvmMaxMemory, long offHeapMaxMemory,
                    long[] deviceTotalMemory, String[] deviceDescription, String hardwareUID);


    /**
     * Report the model information
     *
     * @param modelClassName  Model class name: i.e., type of model
     * @param modelConfigJson Model configuration, as JSON string
     * @param numLayers       Number of layers in the model
     * @param numParams       Number of parameters in the model
     */
    void reportModelInfo(String modelClassName, String modelConfigJson, String[] paramNames, int numLayers,
                    long numParams);


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

    String getSwHostName();

    String getSwJvmUID();

    Map<String, String> getSwEnvironmentInfo();

    int getHwJvmAvailableProcessors();

    int getHwNumDevices();

    long getHwJvmMaxMemory();

    long getHwOffHeapMaxMemory();

    long[] getHwDeviceTotalMemory();

    String[] getHwDeviceDescription();

    String getHwHardwareUID();

    String getModelClassName();

    String getModelConfigJson();

    String[] getModelParamNames();

    int getModelNumLayers();

    long getModelNumParams();

}
