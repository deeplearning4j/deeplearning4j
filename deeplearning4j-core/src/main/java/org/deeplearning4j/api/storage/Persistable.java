package org.deeplearning4j.api.storage;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;

/**
 * Created by Alex on 07/10/2016.
 */
public interface Persistable extends Serializable {

    String getSessionID();

    String getTypeID();

    String getWorkerID();

    long getTimeStamp();


    //SerDe methods:

    /**
     * Length of the encoding, in bytes, when using {@link #encode()}
     * Length may be different using {@link #encode(OutputStream)}, due to things like stream headers
     * @return
     */
    int encodingLengthBytes();

    byte[] encode();

    void encode(ByteBuffer buffer);

    void encode(OutputStream outputStream) throws IOException;

    void decode(byte[] decode);

    void decode(ByteBuffer buffer);

    void decode(InputStream inputStream) throws IOException;

}
