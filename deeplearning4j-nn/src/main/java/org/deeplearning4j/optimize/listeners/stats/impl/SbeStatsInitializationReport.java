package org.deeplearning4j.optimize.listeners.stats.impl;

import lombok.Data;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.deeplearning4j.optimize.listeners.stats.api.StatsInitializationReport;
import org.deeplearning4j.optimize.listeners.stats.sbe.MessageHeaderEncoder;
import org.deeplearning4j.optimize.listeners.stats.sbe.StaticInfoEncoder;

import java.nio.charset.Charset;

/**
 * Created by Alex on 01/10/2016.
 */
@Data
public class SbeStatsInitializationReport implements StatsInitializationReport {

    private static Charset UTF8 = Charset.forName("UTF-8");
    private static byte[] EMPTY_STRING_BYTES = "".getBytes(UTF8);
    private static byte[] EMPTY_BYTES = new byte[0];

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

    private int hwJvmAvailableProcessors = -1;
    private int hwNumDevices = -1;
    private long hwJvmMaxMemory;
    private long hwOffHeapMaxMemory;
    private long[] hwDeviceTotalMemory;
    private String[] hwDeviceDescription;

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
                                   long[] deviceTotalMemory, String[] deviceDescription) {
        this.hwJvmAvailableProcessors = jvmAvailableProcessors;
        this.hwNumDevices = numDevices;
        this.hwJvmMaxMemory = jvmMaxMemory;
        this.hwOffHeapMaxMemory = offHeapMaxMemory;
        this.hwDeviceTotalMemory = deviceTotalMemory;
        this.hwDeviceDescription = deviceDescription;
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

        //TODO we could do this much more efficiently, with buffer re-use, etc.
        MessageHeaderEncoder enc = new MessageHeaderEncoder();
        StaticInfoEncoder sie = new StaticInfoEncoder();


        //First: need to determine how large a buffer to use
        //Fixed length + repeating groups + variable length...
        int bufferSize = 8 + sie.sbeBlockLength();  //Header: 4 x uint16; block length is number of fixed size fields

        //For variable length fields: easist way is simply to convert to UTF-8
        //Of course, it is possible to calculate it first - but we might as well convert, rather than count then convert
        byte[] bswArch = toBytes(hasSoftwareInfo, swArch);
        byte[] bswOsName = toBytes(hasSoftwareInfo, swOsName);
        byte[] bswJvmName = toBytes(hasSoftwareInfo, swJvmName);
        byte[] bswJvmVersion = toBytes(hasSoftwareInfo, swJvmVersion);
        byte[] bswJvmSpecVersion = toBytes(hasSoftwareInfo, swJvmSpecVersion);
        byte[] bswNd4jBackendClass = toBytes(hasSoftwareInfo, swNd4jBackendClass);
        byte[] bswNd4jDataTypeName = toBytes(hasSoftwareInfo, swNd4jDataTypeName);
        byte[] bmodelConfigClass = toBytes(hasModelInfo, modelClassName);
        byte[] bmodelConfigJson = toBytes(hasModelInfo, modelConfigJson);

        byte[][] bhwDeviceDescription = toBytes(hasHardwareInfo, hwDeviceDescription);

        if (hasSoftwareInfo) {
            bufferSize += length(bswArch);
            bufferSize += length(bswOsName);
            bufferSize += length(bswJvmName);
            bufferSize += length(bswJvmVersion);
            bufferSize += length(bswJvmSpecVersion);
            bufferSize += length(bswNd4jBackendClass);
            bufferSize += length(bswNd4jDataTypeName);
        }
        if (hasHardwareInfo) {
            //Device info group:
            bufferSize += (hwDeviceTotalMemory == null ? 0 : hwDeviceTotalMemory.length * 8);
            bufferSize += length(bhwDeviceDescription);
        }

        byte[] bytes = new byte[bufferSize];
        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);

        enc.wrap(buffer, 0)
                .blockLength(sie.sbeBlockLength())
                .templateId(sie.sbeTemplateId())
                .schemaId(sie.sbeSchemaId())
                .version(sie.sbeSchemaVersion());

        int offset = enc.encodedLength();

        //Fixed length fields: always encoded, whether present or not.
        sie.wrap(buffer, offset)
                .time(-1L)  //TODO
                .fieldsPresent()
                .softwareInfo(hasSoftwareInfo)
                .hardwareInfo(hasHardwareInfo)
                .modelInfo(hasModelInfo);
        sie.hwJvmProcessors(hwJvmAvailableProcessors)
                .hwNumDevices((short) hwNumDevices)
                .hwJvmMaxMemory(hwJvmMaxMemory)
                .hwOffheapMaxMemory(hwOffHeapMaxMemory)
                .modelNumLayers(modelNumLayers)
                .modelNumParams(modelNumParams);
        //Device info group...
        StaticInfoEncoder.HwDeviceInfoGroupEncoder hwdEnc = sie.hwDeviceInfoGroupCount(hwNumDevices);
        for(int i=0; i<hwNumDevices; i++ ){
            long maxMem = hwDeviceTotalMemory == null || hwDeviceTotalMemory.length <= i ? -1 : hwDeviceTotalMemory[i];
            byte[] descr = bswNd4jDataTypeName == null || bhwDeviceDescription.length <= i ? EMPTY_STRING_BYTES : bhwDeviceDescription[i];
            if(descr == null) descr = EMPTY_STRING_BYTES;
            hwdEnc.next().deviceMemoryMax(maxMem).putDeviceDescription(descr,0,descr.length);
        }

        //In the case of !hasSoftwareInfo: these will all be empty byte arrays... still need to encode them (for 0 length) however
        sie.putSwArch(bswArch, 0, bswArch.length)
                .putSwOsName(bswOsName, 0, bswOsName.length)
                .putSwJvmName(bswJvmName, 0, bswJvmName.length)
                .putSwJvmVersion(bswJvmVersion, 0, bswJvmVersion.length)
                .putSwJvmSpecVersion(bswJvmSpecVersion, 0, bswJvmSpecVersion.length)
                .putSwNd4jBackendClass(bswNd4jBackendClass, 0, bswNd4jBackendClass.length)
                .putSwNd4jDataTypeName(bswNd4jDataTypeName, 0, bswNd4jDataTypeName.length);
        //Similar: !hasModelInfo -> empty byte[]
        sie.putModelConfigClassName(bmodelConfigClass,0,bmodelConfigClass.length)
                .putModelConfigJson(bmodelConfigJson,0,bmodelConfigJson.length);

        offset += sie.encodedLength();
        if(offset != bytes.length) throw new RuntimeException();

        return bytes;
    }

    private static int length(byte[] bytes) {
        if (bytes == null) return 0;
        return bytes.length;
    }

    private static int length(byte[][] bytes) {
        if (bytes == null) return 0;
        int count = 0;
        for (int i = 0; i < bytes.length; i++) {
            if (bytes[i] != null) count += bytes[i].length;
        }
        return count;
    }

    private static int length(String str) {
        if (str == null) return 0;
        return str.length();
    }

    private static int length(String[] arr) {
        if (arr == null || arr.length == 0) return 0;
        int sum = 0;
        for (String s : arr) sum += length(s);
        return sum;
    }

    private byte[] toBytes(boolean present, String str) {
        if (present || str == null) return EMPTY_BYTES;
        return str.getBytes(UTF8);
    }

    private byte[][] toBytes(boolean present, String[] str) {
        if (str == null) return null;
        byte[][] b = new byte[str.length][0];
        for (int i = 0; i < str.length; i++) {
            if (str[i] == null) continue;
            b[i] = toBytes(present, str[i]);
        }
        return b;
    }

    @Override
    public void fromByteArray(byte[] bytes) {

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
}
