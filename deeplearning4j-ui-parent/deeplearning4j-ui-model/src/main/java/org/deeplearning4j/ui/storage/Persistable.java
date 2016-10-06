package org.deeplearning4j.ui.storage;

/**
 * Created by Alex on 07/10/2016.
 */
public interface Persistable {

    String getSessionID();

    String getTypeID();

    String getWorkerID();


    //SerDe methods:
    int encodingLength();

    byte[] encode();

    void encode(Object buffer); //TODO: type

    byte[] encodeContent();

    void encodeContent(Object buffer);  //TODO: type

    void decode(byte[] decode);

    void decode(Object buffer); //TODO: type

}
