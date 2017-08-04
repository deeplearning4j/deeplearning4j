package org.deeplearning4j.ui.stats.impl.java;

import lombok.Data;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * A 'pure java' implementation of {@link org.deeplearning4j.ui.stats.api.StatsInitializationReport}, mainly used for
 * Java 7 compatibility
 */
@Data
public class JavaStatsInitializationReport implements StatsInitializationReport {

    private String sessionID;
    private String typeID;
    private String workerID;
    private long timeStamp;

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
    private String swHostName;
    private String swJvmUID;
    private Map<String, String> swEnvironmentInfo;

    private int hwJvmAvailableProcessors;
    private int hwNumDevices;
    private long hwJvmMaxMemory;
    private long hwOffHeapMaxMemory;
    private long[] hwDeviceTotalMemory;
    private String[] hwDeviceDescription;
    private String hwHardwareUID;

    private String modelClassName;
    private String modelConfigJson;
    private String[] modelParamNames;
    private int modelNumLayers;
    private long modelNumParams;

    @Override
    public void reportIDs(String sessionID, String typeID, String workerID, long timeStamp) {
        this.sessionID = sessionID;
        this.typeID = typeID;
        this.workerID = workerID;
        this.timeStamp = timeStamp;
    }

    @Override
    public void reportSoftwareInfo(String arch, String osName, String jvmName, String jvmVersion, String jvmSpecVersion,
                    String nd4jBackendClass, String nd4jDataTypeName, String hostname, String jvmUid,
                    Map<String, String> swEnvironmentInfo) {
        this.swArch = arch;
        this.swOsName = osName;
        this.swJvmName = jvmName;
        this.swJvmVersion = jvmVersion;
        this.swJvmSpecVersion = jvmSpecVersion;
        this.swNd4jBackendClass = nd4jBackendClass;
        this.swNd4jDataTypeName = nd4jDataTypeName;
        this.swHostName = hostname;
        this.swJvmUID = jvmUid;
        this.swEnvironmentInfo = swEnvironmentInfo;
        hasSoftwareInfo = true;
    }

    @Override
    public void reportHardwareInfo(int jvmAvailableProcessors, int numDevices, long jvmMaxMemory, long offHeapMaxMemory,
                    long[] deviceTotalMemory, String[] deviceDescription, String hardwareUID) {
        this.hwJvmAvailableProcessors = jvmAvailableProcessors;
        this.hwNumDevices = numDevices;
        this.hwJvmMaxMemory = jvmMaxMemory;
        this.hwOffHeapMaxMemory = offHeapMaxMemory;
        this.hwDeviceTotalMemory = deviceTotalMemory;
        this.hwDeviceDescription = deviceDescription;
        this.hwHardwareUID = hardwareUID;
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
    public boolean hasSoftwareInfo() {
        return hasSoftwareInfo;
    }

    @Override
    public boolean hasHardwareInfo() {
        return hasHardwareInfo;
    }

    @Override
    public boolean hasModelInfo() {
        return hasModelInfo;
    }


    @Override
    public int encodingLengthBytes() {
        //TODO - presumably a more efficient way to do this
        byte[] encoded = encode();
        return encoded.length;
    }

    @Override
    public byte[] encode() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e); //Should never happen
        }
        return baos.toByteArray();
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(outputStream)) {
            oos.writeObject(this);
        }
    }

    @Override
    public void decode(byte[] decode) {
        JavaStatsInitializationReport r;
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decode))) {
            r = (JavaStatsInitializationReport) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e); //Should never happen
        }

        Field[] fields = JavaStatsInitializationReport.class.getDeclaredFields();
        for (Field f : fields) {
            f.setAccessible(true);
            try {
                f.set(this, f.get(r));
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e); //Should never happen
            }
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        decode(bytes);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        decode(IOUtils.toByteArray(inputStream));
    }
}
