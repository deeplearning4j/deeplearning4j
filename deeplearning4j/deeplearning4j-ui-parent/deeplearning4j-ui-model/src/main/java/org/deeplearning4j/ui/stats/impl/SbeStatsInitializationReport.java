/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.stats.impl;

import lombok.Data;
import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.sbe.*;
import org.deeplearning4j.ui.storage.AgronaPersistable;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * An implementation of {@link StatsInitializationReport} using Simple Binary Encoding (SBE)
 *
 * @author Alex Black
 */
@Data
public class SbeStatsInitializationReport implements StatsInitializationReport, AgronaPersistable {

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



    private void clearHwFields() {
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
        swHostName = null;
        swJvmUID = null;
    }

    private void clearModelFields() {
        modelClassName = null;
        modelConfigJson = null;
        modelParamNames = null;
    }

    @Override
    public String getSessionID() {
        return sessionID;
    }

    @Override
    public String getTypeID() {
        return typeID;
    }

    @Override
    public String getWorkerID() {
        return workerID;
    }

    @Override
    public long getTimeStamp() {
        return timeStamp;
    }

    @Override
    public int encodingLengthBytes() {
        //TODO reuse the byte[]s here, to avoid converting them twice...

        //First: need to determine how large a buffer to use.
        //Buffer is composed of:
        //(a) Header: 8 bytes (4x uint16 = 8 bytes)
        //(b) Fixed length entries length (sie.BlockLength())
        //(c) Group 1: Hardware devices (GPUs) max memory: 4 bytes header + nEntries * 8 (int64) + nEntries * variable length Strings (header + content)  = 4 + 8*n + content
        //(d) Group 2: Software device info: 4 bytes header + 2x variable length Strings for each
        //(d) Group 3: Parameter names: 4 bytes header + nEntries * variable length strings (header + content) = 4 + content
        //(e) Variable length fields: 15 String length fields. Size: 4 bytes header, plus content. 60 bytes header
        //Fixed length + repeating groups + variable length...
        StaticInfoEncoder sie = new StaticInfoEncoder();
        int bufferSize = 8 + sie.sbeBlockLength() + 4 + 4 + 60; //header + fixed values + group headers + variable length headers

        //For variable length field lengths: easist way is simply to convert to UTF-8
        //Of course, it is possible to calculate it first - but we might as well convert (1 pass), rather than count then convert (2 passes)
        byte[] bSessionId = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeId = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerId = SbeUtil.toBytes(true, workerID);

        byte[] bswArch = SbeUtil.toBytes(hasSoftwareInfo, swArch);
        byte[] bswOsName = SbeUtil.toBytes(hasSoftwareInfo, swOsName);
        byte[] bswJvmName = SbeUtil.toBytes(hasSoftwareInfo, swJvmName);
        byte[] bswJvmVersion = SbeUtil.toBytes(hasSoftwareInfo, swJvmVersion);
        byte[] bswJvmSpecVersion = SbeUtil.toBytes(hasSoftwareInfo, swJvmSpecVersion);
        byte[] bswNd4jBackendClass = SbeUtil.toBytes(hasSoftwareInfo, swNd4jBackendClass);
        byte[] bswNd4jDataTypeName = SbeUtil.toBytes(hasSoftwareInfo, swNd4jDataTypeName);
        byte[] bswHostname = SbeUtil.toBytes(hasSoftwareInfo, swHostName);
        byte[] bswJvmUID = SbeUtil.toBytes(hasSoftwareInfo, swJvmUID);
        byte[] bHwHardwareUID = SbeUtil.toBytes(hasHardwareInfo, hwHardwareUID);
        byte[] bmodelConfigClass = SbeUtil.toBytes(hasModelInfo, modelClassName);
        byte[] bmodelConfigJson = SbeUtil.toBytes(hasModelInfo, modelConfigJson);

        byte[][] bhwDeviceDescription = SbeUtil.toBytes(hasHardwareInfo, hwDeviceDescription);
        byte[][][] bswEnvInfo = SbeUtil.toBytes(swEnvironmentInfo);
        byte[][] bModelParamNames = SbeUtil.toBytes(hasModelInfo, modelParamNames);



        bufferSize += bSessionId.length + bTypeId.length + bWorkerId.length;

        bufferSize += 4; //swEnvironmentInfo group header (always present)
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
            //For each entry: 2 variable-length headers (2x4 bytes each) + content
            int envCount = (bswEnvInfo != null ? bswEnvInfo.length : 0);
            bufferSize += envCount * 8;
            bufferSize += SbeUtil.length(bswEnvInfo);
        }
        int nHWDeviceStats = hwNumDevices;
        if (!hasHardwareInfo)
            nHWDeviceStats = 0;
        if (hasHardwareInfo) {
            //Device info group:
            bufferSize += hwNumDevices * 8; //fixed content in group: int64 -> 8 bytes. Encode an entry, even if hwDeviceTotalMemory is null
            bufferSize += hwNumDevices * 4; //uint32: 4 bytes per entry for var length header...; as above
            bufferSize += SbeUtil.length(bhwDeviceDescription);
            bufferSize += SbeUtil.length(bHwHardwareUID);
        }
        if (hasModelInfo) {
            bufferSize += SbeUtil.length(bmodelConfigClass);
            bufferSize += SbeUtil.length(bmodelConfigJson);
            bufferSize += SbeUtil.length(bModelParamNames);
            bufferSize += (bModelParamNames == null ? 0 : bModelParamNames.length * 4); //uint32: 4 bytes per entry for var length header...
        }

        return bufferSize;
    }

    @Override
    public byte[] encode() {
        byte[] bytes = new byte[encodingLengthBytes()];
        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);
        encode(buffer);
        return bytes;
    }

    @Override
    public void encode(ByteBuffer buffer) {
        encode(new UnsafeBuffer(buffer));
    }

    @Override
    public void encode(MutableDirectBuffer buffer) {

        MessageHeaderEncoder enc = new MessageHeaderEncoder();
        StaticInfoEncoder sie = new StaticInfoEncoder();

        byte[] bSessionId = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeId = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerId = SbeUtil.toBytes(true, workerID);

        byte[] bswArch = SbeUtil.toBytes(hasSoftwareInfo, swArch);
        byte[] bswOsName = SbeUtil.toBytes(hasSoftwareInfo, swOsName);
        byte[] bswJvmName = SbeUtil.toBytes(hasSoftwareInfo, swJvmName);
        byte[] bswJvmVersion = SbeUtil.toBytes(hasSoftwareInfo, swJvmVersion);
        byte[] bswJvmSpecVersion = SbeUtil.toBytes(hasSoftwareInfo, swJvmSpecVersion);
        byte[] bswNd4jBackendClass = SbeUtil.toBytes(hasSoftwareInfo, swNd4jBackendClass);
        byte[] bswNd4jDataTypeName = SbeUtil.toBytes(hasSoftwareInfo, swNd4jDataTypeName);
        byte[] bswHostname = SbeUtil.toBytes(hasSoftwareInfo, swHostName);
        byte[] bswJvmUID = SbeUtil.toBytes(hasSoftwareInfo, swJvmUID);
        byte[] bHwHardwareUID = SbeUtil.toBytes(hasHardwareInfo, hwHardwareUID);
        byte[] bmodelConfigClass = SbeUtil.toBytes(hasModelInfo, modelClassName);
        byte[] bmodelConfigJson = SbeUtil.toBytes(hasModelInfo, modelConfigJson);

        byte[][] bhwDeviceDescription = SbeUtil.toBytes(hasHardwareInfo, hwDeviceDescription);
        byte[][][] bswEnvInfo = SbeUtil.toBytes(swEnvironmentInfo);
        byte[][] bModelParamNames = SbeUtil.toBytes(hasModelInfo, modelParamNames);

        enc.wrap(buffer, 0).blockLength(sie.sbeBlockLength()).templateId(sie.sbeTemplateId())
                        .schemaId(sie.sbeSchemaId()).version(sie.sbeSchemaVersion());

        int offset = enc.encodedLength(); //Expect 8 bytes...

        //Fixed length fields: always encoded, whether present or not.
        sie.wrap(buffer, offset).time(timeStamp).fieldsPresent().softwareInfo(hasSoftwareInfo)
                        .hardwareInfo(hasHardwareInfo).modelInfo(hasModelInfo);
        sie.hwJvmProcessors(hwJvmAvailableProcessors).hwNumDevices((short) hwNumDevices).hwJvmMaxMemory(hwJvmMaxMemory)
                        .hwOffheapMaxMemory(hwOffHeapMaxMemory).modelNumLayers(modelNumLayers)
                        .modelNumParams(modelNumParams);
        //Device info group...
        StaticInfoEncoder.HwDeviceInfoGroupEncoder hwdEnc = sie.hwDeviceInfoGroupCount(hwNumDevices);
        int nHWDeviceStats = (hasHardwareInfo ? hwNumDevices : 0);
        for (int i = 0; i < nHWDeviceStats; i++) {
            long maxMem = hwDeviceTotalMemory == null || hwDeviceTotalMemory.length <= i ? 0 : hwDeviceTotalMemory[i];
            byte[] descr = bhwDeviceDescription == null || bhwDeviceDescription.length <= i ? SbeUtil.EMPTY_BYTES
                            : bhwDeviceDescription[i];
            if (descr == null)
                descr = SbeUtil.EMPTY_BYTES;
            hwdEnc.next().deviceMemoryMax(maxMem).putDeviceDescription(descr, 0, descr.length);
        }

        //Environment info group
        int numEnvValues = (hasSoftwareInfo && swEnvironmentInfo != null ? swEnvironmentInfo.size() : 0);
        StaticInfoEncoder.SwEnvironmentInfoEncoder swEnv = sie.swEnvironmentInfoCount(numEnvValues);
        if (numEnvValues > 0) {
            byte[][][] mapAsBytes = SbeUtil.toBytes(swEnvironmentInfo);
            for (byte[][] entryBytes : mapAsBytes) {
                swEnv.next().putEnvKey(entryBytes[0], 0, entryBytes[0].length).putEnvValue(entryBytes[1], 0,
                                entryBytes[1].length);
            }
        }

        int nParamNames = modelParamNames == null ? 0 : modelParamNames.length;
        StaticInfoEncoder.ModelParamNamesEncoder mpnEnc = sie.modelParamNamesCount(nParamNames);
        for (int i = 0; i < nParamNames; i++) {
            mpnEnc.next().putModelParamNames(bModelParamNames[i], 0, bModelParamNames[i].length);
        }

        //In the case of !hasSoftwareInfo: these will all be empty byte arrays... still need to encode them (for 0 length) however
        sie.putSessionID(bSessionId, 0, bSessionId.length).putTypeID(bTypeId, 0, bTypeId.length)
                        .putWorkerID(bWorkerId, 0, bWorkerId.length).putSwArch(bswArch, 0, bswArch.length)
                        .putSwOsName(bswOsName, 0, bswOsName.length).putSwJvmName(bswJvmName, 0, bswJvmName.length)
                        .putSwJvmVersion(bswJvmVersion, 0, bswJvmVersion.length)
                        .putSwJvmSpecVersion(bswJvmSpecVersion, 0, bswJvmSpecVersion.length)
                        .putSwNd4jBackendClass(bswNd4jBackendClass, 0, bswNd4jBackendClass.length)
                        .putSwNd4jDataTypeName(bswNd4jDataTypeName, 0, bswNd4jDataTypeName.length)
                        .putSwHostName(bswHostname, 0, bswHostname.length).putSwJvmUID(bswJvmUID, 0, bswJvmUID.length)
                        .putHwHardwareUID(bHwHardwareUID, 0, bHwHardwareUID.length);
        //Similar: !hasModelInfo -> empty byte[]
        sie.putModelConfigClassName(bmodelConfigClass, 0, bmodelConfigClass.length).putModelConfigJson(bmodelConfigJson,
                        0, bmodelConfigJson.length);
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        //TODO there may be more efficient way of doing this
        outputStream.write(encode());
    }

    @Override
    public void decode(byte[] decode) {
        MutableDirectBuffer buffer = new UnsafeBuffer(decode);
        decode(buffer);
    }

    @Override
    public void decode(ByteBuffer buffer) {
        decode(new UnsafeBuffer(buffer));
    }

    @Override
    public void decode(DirectBuffer buffer) {
        //TODO we could do this much more efficiently, with buffer re-use, etc.
        MessageHeaderDecoder dec = new MessageHeaderDecoder();
        StaticInfoDecoder sid = new StaticInfoDecoder();
        dec.wrap(buffer, 0);

        final int blockLength = dec.blockLength();
        final int version = dec.version();

        final int headerLength = dec.encodedLength();
        //TODO: in general, we should check the header, version, schema etc. But we don't have any other versions yet.

        sid.wrap(buffer, headerLength, blockLength, version);
        timeStamp = sid.time();
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

        //Hardware device info group
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

        //Environment info group
        i = 0;
        StaticInfoDecoder.SwEnvironmentInfoDecoder swEnvDecoder = sid.swEnvironmentInfo();
        if (swEnvDecoder.count() > 0) {
            swEnvironmentInfo = new HashMap<>();
        }
        for (StaticInfoDecoder.SwEnvironmentInfoDecoder env : swEnvDecoder) {
            String key = env.envKey();
            String value = env.envValue();
            swEnvironmentInfo.put(key, value);
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
        sessionID = sid.sessionID();
        typeID = sid.typeID();
        workerID = sid.workerID();
        swArch = sid.swArch();
        swOsName = sid.swOsName();
        swJvmName = sid.swJvmName();
        swJvmVersion = sid.swJvmVersion();
        swJvmSpecVersion = sid.swJvmSpecVersion();
        swNd4jBackendClass = sid.swNd4jBackendClass();
        swNd4jDataTypeName = sid.swNd4jDataTypeName();
        swHostName = sid.swHostName();
        swJvmUID = sid.swJvmUID();
        if (!hasSoftwareInfo)
            clearSwFields();
        hwHardwareUID = sid.hwHardwareUID();
        if (!hasHardwareInfo)
            clearHwFields();
        modelClassName = sid.modelConfigClassName();
        modelConfigJson = sid.modelConfigJson();
        if (!hasModelInfo)
            clearModelFields();
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        byte[] bytes = IOUtils.toByteArray(inputStream);
        decode(bytes);
    }
}
