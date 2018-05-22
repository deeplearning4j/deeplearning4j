package org.deeplearning4j.ui.storage.impl;

import lombok.Data;
import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.stats.impl.SbeUtil;
import org.deeplearning4j.ui.stats.sbe.MessageHeaderDecoder;
import org.deeplearning4j.ui.stats.sbe.MessageHeaderEncoder;
import org.deeplearning4j.ui.stats.sbe.StorageMetaDataDecoder;
import org.deeplearning4j.ui.stats.sbe.StorageMetaDataEncoder;
import org.deeplearning4j.ui.storage.AgronaPersistable;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;

/**
 * SbeStorageMetaData: stores information about a given session: for example, the types of the static and update information.
 *
 * @author Alex Black
 */
@Data
public class SbeStorageMetaData implements org.deeplearning4j.api.storage.StorageMetaData, AgronaPersistable {


    private long timeStamp;
    private String sessionID;
    private String typeID;
    private String workerID;
    private String initTypeClass;
    private String updateTypeClass;
    //Store serialized; saves class exceptions if we don't have the right class, and don't care about deserializing
    // on this machine, right now
    private byte[] extraMeta;

    public SbeStorageMetaData() {
        //No arg constructor for serialization/deserialization
    }

    public SbeStorageMetaData(long timeStamp, String sessionID, String typeID, String workerID, Class<?> initType,
                    Class<?> updateType) {
        this(timeStamp, sessionID, typeID, workerID, (initType != null ? initType.getName() : null),
                        (updateType != null ? updateType.getName() : null));
    }

    public SbeStorageMetaData(long timeStamp, String sessionID, String typeID, String workerID, String initTypeClass,
                    String updateTypeClass) {
        this(timeStamp, sessionID, typeID, workerID, initTypeClass, updateTypeClass, null);
    }

    public SbeStorageMetaData(long timeStamp, String sessionID, String typeID, String workerID, String initTypeClass,
                    String updateTypeClass, Serializable extraMetaData) {
        this.timeStamp = timeStamp;
        this.sessionID = sessionID;
        this.typeID = typeID;
        this.workerID = workerID;
        this.initTypeClass = initTypeClass;
        this.updateTypeClass = updateTypeClass;
        this.extraMeta = (extraMetaData == null ? null : SbeUtil.toBytesSerializable(extraMetaData));
    }

    public Serializable getExtraMetaData() {
        return SbeUtil.fromBytesSerializable(extraMeta);
    }

    @Override
    public int encodingLengthBytes() {
        //TODO store byte[]s so we don't end up calculating again in encode
        //SBE buffer is composed of:
        //(a) Header: 8 bytes (4x uint16 = 8 bytes)
        //(b) timestamp: fixed length long value (8 bytes)
        //(b) 5 variable length fields. 4 bytes header (each) + content = 20 bytes + content
        //(c) Variable length byte[]. 4 bytes header + content

        int bufferSize = 8 + 8 + 20 + 4;
        byte[] bSessionID = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeID = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerID = SbeUtil.toBytes(true, workerID);
        byte[] bInitTypeClass = SbeUtil.toBytes(true, initTypeClass);
        byte[] bUpdateTypeClass = SbeUtil.toBytes(true, updateTypeClass);
        byte[] bExtraMetaData = SbeUtil.toBytesSerializable(extraMeta);

        bufferSize += bSessionID.length + bTypeID.length + bWorkerID.length + bInitTypeClass.length
                        + bUpdateTypeClass.length + bExtraMetaData.length;

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
        StorageMetaDataEncoder smde = new StorageMetaDataEncoder();

        enc.wrap(buffer, 0).blockLength(smde.sbeBlockLength()).templateId(smde.sbeTemplateId())
                        .schemaId(smde.sbeSchemaId()).version(smde.sbeSchemaVersion());

        int offset = enc.encodedLength(); //Expect 8 bytes

        byte[] bSessionID = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeID = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerID = SbeUtil.toBytes(true, workerID);
        byte[] bInitTypeClass = SbeUtil.toBytes(true, initTypeClass);
        byte[] bUpdateTypeClass = SbeUtil.toBytes(true, updateTypeClass);

        smde.wrap(buffer, offset).timeStamp(timeStamp);

        StorageMetaDataEncoder.ExtraMetaDataBytesEncoder ext =
                        smde.extraMetaDataBytesCount(extraMeta == null ? 0 : extraMeta.length);
        if (extraMeta != null) {
            for (byte b : extraMeta) {
                ext.next().bytes(b);
            }
        }

        smde.putSessionID(bSessionID, 0, bSessionID.length).putTypeID(bTypeID, 0, bTypeID.length)
                        .putWorkerID(bWorkerID, 0, bWorkerID.length)
                        .putInitTypeClass(bInitTypeClass, 0, bInitTypeClass.length)
                        .putUpdateTypeClass(bUpdateTypeClass, 0, bUpdateTypeClass.length);
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

        MessageHeaderDecoder dec = new MessageHeaderDecoder();
        dec.wrap(buffer, 0);
        final int blockLength = dec.blockLength();
        final int version = dec.version();
        final int headerLength = dec.encodedLength();

        //TODO Validate header, version etc

        StorageMetaDataDecoder smdd = new StorageMetaDataDecoder();
        smdd.wrap(buffer, headerLength, blockLength, version);
        timeStamp = smdd.timeStamp();

        StorageMetaDataDecoder.ExtraMetaDataBytesDecoder ext = smdd.extraMetaDataBytes();
        int length = ext.count();
        if (length > 0) {
            extraMeta = new byte[length];
            int i = 0;
            for (StorageMetaDataDecoder.ExtraMetaDataBytesDecoder d : ext) {
                extraMeta[i++] = d.bytes();
            }
        }

        sessionID = smdd.sessionID();
        typeID = smdd.typeID();
        workerID = smdd.workerID();
        initTypeClass = smdd.initTypeClass();
        updateTypeClass = smdd.updateTypeClass();
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        byte[] bytes = IOUtils.toByteArray(inputStream);
        decode(bytes);
    }
}
