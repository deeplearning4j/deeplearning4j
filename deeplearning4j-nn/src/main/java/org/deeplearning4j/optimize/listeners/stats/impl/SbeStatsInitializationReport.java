package org.deeplearning4j.optimize.listeners.stats.impl;

import lombok.Data;
import org.deeplearning4j.optimize.listeners.stats.api.StatsInitializationReport;

/**
 * Created by Alex on 01/10/2016.
 */
@Data
public class SbeStatsInitializationReport implements StatsInitializationReport {

    private boolean hasSoftwareInfo;
    private boolean hasHardwareInfo;
    private boolean hasModelInfo;

    private String swArch;
    private String swOsName;
    private String swJvmName;
    private String swJvmVersion;
    private String swJvmSpecVersion;
    private String swNd4jBackendClass;
    private String swNd4jDataTypeName;

    private int hwJvmAvailableProcessors;
    private int hwNumDevices;
    private Long hwJvmMaxMemory;
    private Long hwOffHeapMaxMemory;
    private long[] hwDeviceTotalMemory;

    private String modelClassName;
    private String modelConfigJson;
    private String[] modelParamNames;
    private int modelNumLayers;
    private long modelNumParams;


    @Override
    public void reportSoftwareInfo(String arch, String osName, String jvmName, String jvmVersion, String jvmSpecVersion,
                                   String nd4jBackendClass, String nd4jDataTypeName) {
        this.swArch = arch;
        this.swOsName = osName;
        this.swJvmName = jvmName;
        this.swJvmVersion = jvmVersion;
        this.swJvmSpecVersion = jvmSpecVersion;
        this.swNd4jBackendClass = nd4jBackendClass;
        this.swNd4jDataTypeName = nd4jDataTypeName;
        hasSoftwareInfo = true;
    }

    @Override
    public void reportHardwareInfo(int jvmAvailableProcessors, int numDevices, long jvmMaxMemory, long offHeapMaxMemory,
                                   long[] deviceTotalMemory) {
        this.hwJvmAvailableProcessors = jvmAvailableProcessors;
        this.hwNumDevices = numDevices;
        this.hwJvmMaxMemory = jvmMaxMemory;
        this.hwOffHeapMaxMemory = offHeapMaxMemory;
        this.hwDeviceTotalMemory = deviceTotalMemory;
        hasHardwareInfo = true;
    }

    @Override
    public void reportModelInfo(String modelClassName, String modelConfigJson, String[] modelParamNames, int numLayers,
                                long numParams) {
        this.modelClassName = modelClassName;
        this.modelConfigJson = modelConfigJson;
        this.modelParamNames = modelParamNames;
        this.modelNumLayers = numLayers;
        this.modelNumParams = numParams;
        hasModelInfo = true;
    }

    @Override
    public byte[] toByteArray() {
        //Recall that the encoding order is VERY important for SBE... must follow the schema exactly

    }

    @Override
    public void fromByteArray(byte[] bytes) {

    }

    @Override
    public boolean hasSoftwareInfo(){
        return hasSoftwareInfo;
    }

    @Override
    public boolean hasHardwareInfo(){
        return hasHardwareInfo;
    }

    @Override
    public boolean hasModelInfo(){
        return hasModelInfo;
    }
}
