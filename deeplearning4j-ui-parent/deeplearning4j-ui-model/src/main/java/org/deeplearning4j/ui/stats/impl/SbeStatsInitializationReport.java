package org.deeplearning4j.ui.stats.impl;

import lombok.Data;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.sbe.*;

/**
 * An implementation of {@link StatsInitializationReport} using Simple Binary Encoding (SBE)
 *
 * @author Alex Black
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
    private String swHostname;
    private String swJvmUID;

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
    public void reportSoftwareInfo(String arch, String osName, String jvmName, String jvmVersion, String jvmSpecVersion,
                                   String nd4jBackendClass, String nd4jDataTypeName, String hostname, String jvmUid) {
        this.swArch = arch;
        this.swOsName = osName;
        this.swJvmName = jvmName;
        this.swJvmVersion = jvmVersion;
        this.swJvmSpecVersion = jvmSpecVersion;
        this.swNd4jBackendClass = nd4jBackendClass;
        this.swNd4jDataTypeName = nd4jDataTypeName;
        this.swHostname = hostname;
        this.swJvmUID = jvmUid;
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
    public byte[] toByteArray() {
        //Recall that the encoding order is VERY important for SBE... must follow the schema exactly

        //TODO we could do this much more efficiently, with buffer re-use, etc.
        MessageHeaderEncoder enc = new MessageHeaderEncoder();
        StaticInfoEncoder sie = new StaticInfoEncoder();


        //First: need to determine how large a buffer to use.
        //Buffer is composed of:
        //(a) Header: 8 bytes (4x uint16 = 8 bytes)
        //(b) Fixed length entries length (sie.BlockLength())
        //(c) Group 1: Hardware devices (GPUs) max memory: 4 bytes header + nEntries * 8 (int64) + nEntries * variable length Strings (header + content)  = 4 + 8*n + content
        //(d) Group 2: Parameter names: 4 bytes header + nEntries * variable length strings (header + content) = 4 + content
        //(e) Variable length fields: 12 String length fields. Size: 4 bytes header, plus content. 48 bytes header
        //Fixed length + repeating groups + variable length...
        int bufferSize = 8 + sie.sbeBlockLength() + 4 + 4 + 48; //header + fixed values + group headers + variable length headers

        //For variable length field lengths: easist way is simply to convert to UTF-8
        //Of course, it is possible to calculate it first - but we might as well convert (1 pass), rather than count then convert (2 passes)
        byte[] bswArch = SbeUtil.toBytes(hasSoftwareInfo, swArch);
        byte[] bswOsName = SbeUtil.toBytes(hasSoftwareInfo, swOsName);
        byte[] bswJvmName = SbeUtil.toBytes(hasSoftwareInfo, swJvmName);
        byte[] bswJvmVersion = SbeUtil.toBytes(hasSoftwareInfo, swJvmVersion);
        byte[] bswJvmSpecVersion = SbeUtil.toBytes(hasSoftwareInfo, swJvmSpecVersion);
        byte[] bswNd4jBackendClass = SbeUtil.toBytes(hasSoftwareInfo, swNd4jBackendClass);
        byte[] bswNd4jDataTypeName = SbeUtil.toBytes(hasSoftwareInfo, swNd4jDataTypeName);
        byte[] bswHostname = SbeUtil.toBytes(hasSoftwareInfo, swHostname);
        byte[] bswJvmUID = SbeUtil.toBytes(hasSoftwareInfo, swJvmUID);
        byte[] bHwHardwareUID = SbeUtil.toBytes(hasHardwareInfo, hwHardwareUID);
        byte[] bmodelConfigClass = SbeUtil.toBytes(hasModelInfo, modelClassName);
        byte[] bmodelConfigJson = SbeUtil.toBytes(hasModelInfo, modelConfigJson);

        byte[][] bhwDeviceDescription = SbeUtil.toBytes(hasHardwareInfo, hwDeviceDescription);
        byte[][] bModelParamNames = SbeUtil.toBytes(hasModelInfo, modelParamNames);

        if (hasSoftwareInfo) {
            bufferSize += SbeUtil.length(bswArch);
            bufferSize += SbeUtil.length(bswOsName);
            bufferSize += SbeUtil.length(bswJvmName);
            bufferSize += SbeUtil.length(bswJvmVersion);
            bufferSize += SbeUtil.length(bswJvmSpecVersion);
            bufferSize += SbeUtil.length(bswNd4jBackendClass);
            bufferSize += SbeUtil.length(bswNd4jDataTypeName);
            bufferSize += SbeUtil.length(bswHostname);
            bufferSize += SbeUtil.length(bswJvmUID);
        }
        int nHWDeviceStats = hwNumDevices;
        if (!hasHardwareInfo) nHWDeviceStats = 0;
        if (hasHardwareInfo) {
            //Device info group:
            bufferSize += hwNumDevices * 8;     //fixed content in group: int64 -> 8 bytes. Encode an entry, even if hwDeviceTotalMemory is null
            bufferSize += hwNumDevices * 4;     //uint32: 4 bytes per entry for var length header...; as above
            bufferSize += SbeUtil.length(bhwDeviceDescription);
            bufferSize += SbeUtil.length(bHwHardwareUID);
        }
        if (hasModelInfo) {
            bufferSize += SbeUtil.length(bmodelConfigClass);
            bufferSize += SbeUtil.length(bmodelConfigJson);
            bufferSize += SbeUtil.length(bModelParamNames);
            bufferSize += (bModelParamNames == null ? 0 : bModelParamNames.length * 4);   //uint32: 4 bytes per entry for var length header...
        }


        //Now know the buffer size -> create appropriate sized byte array
        byte[] bytes = new byte[bufferSize];
        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);

        enc.wrap(buffer, 0)
                .blockLength(sie.sbeBlockLength())
                .templateId(sie.sbeTemplateId())
                .schemaId(sie.sbeSchemaId())
                .version(sie.sbeSchemaVersion());

        int offset = enc.encodedLength();   //Expect 8 bytes...

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
        for (int i = 0; i < nHWDeviceStats; i++) {
            long maxMem = hwDeviceTotalMemory == null || hwDeviceTotalMemory.length <= i ? 0 : hwDeviceTotalMemory[i];
            byte[] descr = bhwDeviceDescription == null || bhwDeviceDescription.length <= i ? SbeUtil.EMPTY_BYTES : bhwDeviceDescription[i];
            if (descr == null) descr = SbeUtil.EMPTY_BYTES;
            hwdEnc.next().deviceMemoryMax(maxMem).putDeviceDescription(descr, 0, descr.length);
        }

        int nParamNames = modelParamNames == null ? 0 : modelParamNames.length;
        StaticInfoEncoder.ModelParamNamesEncoder mpnEnc = sie.modelParamNamesCount(nParamNames);
        for (int i = 0; i < nParamNames; i++) {
            mpnEnc.next().putModelParamNames(bModelParamNames[i], 0, bModelParamNames[i].length);
        }

        //In the case of !hasSoftwareInfo: these will all be empty byte arrays... still need to encode them (for 0 length) however
        sie.putSwArch(bswArch, 0, bswArch.length)
                .putSwOsName(bswOsName, 0, bswOsName.length)
                .putSwJvmName(bswJvmName, 0, bswJvmName.length)
                .putSwJvmVersion(bswJvmVersion, 0, bswJvmVersion.length)
                .putSwJvmSpecVersion(bswJvmSpecVersion, 0, bswJvmSpecVersion.length)
                .putSwNd4jBackendClass(bswNd4jBackendClass, 0, bswNd4jBackendClass.length)
                .putSwNd4jDataTypeName(bswNd4jDataTypeName, 0, bswNd4jDataTypeName.length)
                .putSwHostName(bswHostname, 0, bswHostname.length)
                .putSwJvmUID(bswJvmUID, 0, bswJvmUID.length)
                .putHwHardwareUID(bHwHardwareUID, 0, bHwHardwareUID.length);
        //Similar: !hasModelInfo -> empty byte[]
        sie.putModelConfigClassName(bmodelConfigClass, 0, bmodelConfigClass.length)
                .putModelConfigJson(bmodelConfigJson, 0, bmodelConfigJson.length);

        offset += sie.encodedLength();
        if (offset != bytes.length) {
            throw new RuntimeException();
        }

        return bytes;
    }


    @Override
    public void fromByteArray(byte[] bytes) {
        //TODO we could do this much more efficiently, with buffer re-use, etc.
        MessageHeaderDecoder dec = new MessageHeaderDecoder();
        StaticInfoDecoder sid = new StaticInfoDecoder();

        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);
        dec.wrap(buffer, 0);

        final int blockLength = dec.blockLength();
        final int version = dec.version();

        int headerLength = dec.encodedLength();
        //TODO: in general, we should check the header, version, schema etc. But we don't have any other versions yet.

        sid.wrap(buffer, headerLength, blockLength, version);
        long time = sid.time(); //TODO
        InitFieldsPresentDecoder fields = sid.fieldsPresent();
        hasSoftwareInfo = fields.softwareInfo();
        hasHardwareInfo = fields.hardwareInfo();
        hasModelInfo = fields.modelInfo();

        //These fields: always present, even if !hasHardwareInfo
        hwJvmAvailableProcessors = sid.hwJvmProcessors();
        hwNumDevices = sid.hwNumDevices();
        hwJvmMaxMemory = sid.hwJvmMaxMemory();
        hwOffHeapMaxMemory = sid.hwOffheapMaxMemory();
        modelNumLayers = sid.modelNumLayers();
        modelNumParams = sid.modelNumParams();

        StaticInfoDecoder.HwDeviceInfoGroupDecoder hwDeviceInfoGroupDecoder = sid.hwDeviceInfoGroup();
        int count = hwDeviceInfoGroupDecoder.count();
        if (count > 0) {
            hwDeviceTotalMemory = new long[count];
            hwDeviceDescription = new String[count];
        }
        int i = 0;
        for (StaticInfoDecoder.HwDeviceInfoGroupDecoder hw : hwDeviceInfoGroupDecoder) {
            hwDeviceTotalMemory[i] = hw.deviceMemoryMax();
            hwDeviceDescription[i++] = hw.deviceDescription();
        }

        i = 0;
        StaticInfoDecoder.ModelParamNamesDecoder mpdec = sid.modelParamNames();
        int mpnCount = mpdec.count();
        modelParamNames = new String[mpnCount];
        for (StaticInfoDecoder.ModelParamNamesDecoder mp : mpdec) {
            modelParamNames[i++] = mp.modelParamNames();
        }
        //Variable length data. Even if it is missing: still needs to be read, to advance buffer
        //Again, the exact order of these calls matters here
        swArch = sid.swArch();
        swOsName = sid.swOsName();
        swJvmName = sid.swJvmName();
        swJvmVersion = sid.swJvmVersion();
        swJvmSpecVersion = sid.swJvmSpecVersion();
        swNd4jBackendClass = sid.swNd4jBackendClass();
        swNd4jDataTypeName = sid.swNd4jDataTypeName();
        swHostname = sid.swHostName();
        swJvmUID = sid.swJvmUID();
        if (!hasSoftwareInfo) clearSwFields();
        hwHardwareUID = sid.hwHardwareUID();
        if(!hasHardwareInfo) clearHwFields();
        modelClassName = sid.modelConfigClassName();
        modelConfigJson = sid.modelConfigJson();
        if (!hasModelInfo) clearModelFields();
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

    private void clearHwFields(){
        hwDeviceTotalMemory = null;
        hwDeviceDescription = null;
        hwHardwareUID = null;
    }

    private void clearSwFields() {
        swArch = null;
        swOsName = null;
        swJvmName = null;
        swJvmVersion = null;
        swJvmSpecVersion = null;
        swNd4jBackendClass = null;
        swNd4jDataTypeName = null;
        swHostname = null;
        swJvmUID = null;
    }

    private void clearModelFields() {
        modelClassName = null;
        modelConfigJson = null;
        modelParamNames = null;
    }
}
