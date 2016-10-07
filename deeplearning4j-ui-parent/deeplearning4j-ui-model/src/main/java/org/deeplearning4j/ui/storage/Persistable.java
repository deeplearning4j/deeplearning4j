package org.deeplearning4j.ui.storage;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;

/**
 * Created by Alex on 07/10/2016.
 */
public interface Persistable extends Serializable {

    String getSessionID();

    String getTypeID();

    String getWorkerID();


    //SerDe methods:

    /**
     * Length of the encoding, in bytes
     * @return
     */
    int encodingLength();

    byte[] encode();

    void encode(Object buffer); //TODO: type

    void encode(OutputStream outputStream);

//    byte[] encodeContent();
//
//    void encodeContent(Object buffer);  //TODO: type

    void decode(byte[] decode);

    void decode(Object buffer); //TODO: type

    void decode(InputStream inputStream);

}
