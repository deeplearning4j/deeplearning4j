package org.deeplearning4j.ui.storage;

import lombok.AllArgsConstructor;
import lombok.Data;

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
    public int encodingLength() {

        return -1;
    }

    @Override
    public byte[] encode() {
        return new byte[0];
    }

    @Override
    public void encode(Object buffer) {

    }

    @Override
    public void encode(OutputStream outputStream) {

    }

    @Override
    public void decode(byte[] decode) {

    }

    @Override
    public void decode(Object buffer) {

    }

    @Override
    public void decode(InputStream inputStream) {

    }
}
