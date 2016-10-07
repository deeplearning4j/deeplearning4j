package org.deeplearning4j.ui.storage;

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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Created by Alex on 07/10/2016.
 */
@Data
public class StorageMetaData implements Persistable {


    private String sessionID;
    private String typeID;
    private String workerID;
    private String initTypeClass;
    private String updateTypeClass;

    public StorageMetaData(){
        //No arg constructor for serialization/deserialization
    }

    public StorageMetaData(String sessionID, String typeID, String workerID, Class<?> initType, Class<?> updateType){
        this(sessionID, typeID, workerID, (initType != null ? initType.getName() : null),
                (updateType != null ? updateType.getName() : null));
    }

    public StorageMetaData(String sessionID, String typeID, String workerID, String initTypeClass, String updateTypeClass ){
        this.sessionID = sessionID;
        this.typeID = typeID;
        this.workerID = workerID;
        this.initTypeClass = initTypeClass;
        this.updateTypeClass = updateTypeClass;
    }

    @Override
    public int encodingLengthBytes() {
        //TODO store byte[]s so we don't end up calculating again in encode
        //SBE buffer is composed of:
        //(a) Header: 8 bytes (4x uint16 = 8 bytes)
        //(b) 5 variable length fields. 4 bytes header (each) + content = 20 bytes + content

        int bufferSize = 8 + 20;
        byte[] bSessionID = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeID = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerID = SbeUtil.toBytes(true, workerID);
        byte[] bInitTypeClass = SbeUtil.toBytes(true, initTypeClass);
        byte[] bUpdateTypeClass = SbeUtil.toBytes(true, updateTypeClass);

        bufferSize += bSessionID.length + bTypeID.length + bWorkerID.length + bInitTypeClass.length + bUpdateTypeClass.length;

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
    public void encode(MutableDirectBuffer buffer) {
        MessageHeaderEncoder enc = new MessageHeaderEncoder();
        StorageMetaDataEncoder smde = new StorageMetaDataEncoder();

        enc.wrap(buffer, 0)
                .blockLength(smde.sbeBlockLength())
                .templateId(smde.sbeTemplateId())
                .schemaId(smde.sbeSchemaId())
                .version(smde.sbeSchemaVersion());

        int offset = enc.encodedLength();   //Expect 8 bytes

        byte[] bSessionID = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeID = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerID = SbeUtil.toBytes(true, workerID);
        byte[] bInitTypeClass = SbeUtil.toBytes(true, initTypeClass);
        byte[] bUpdateTypeClass = SbeUtil.toBytes(true, updateTypeClass);

        smde.wrap(buffer, offset)
                .putSessionID(bSessionID, 0, bSessionID.length)
                .putTypeID(bTypeID, 0, bTypeID.length)
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
    public void decode(DirectBuffer buffer) {

        MessageHeaderDecoder dec = new MessageHeaderDecoder();
        dec.wrap(buffer, 0);
        final int blockLength = dec.blockLength();
        final int version = dec.version();
        final int headerLength = dec.encodedLength();

        //TODO Validate header, version etc

        StorageMetaDataDecoder smdd = new StorageMetaDataDecoder();
        smdd.wrap(buffer, headerLength, blockLength, version);

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
