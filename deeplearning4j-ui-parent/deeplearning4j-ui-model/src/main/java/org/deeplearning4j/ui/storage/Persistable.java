package org.deeplearning4j.ui.storage;

import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;

import java.io.IOException;
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

    long getTimeStamp();


    //SerDe methods:

    /**
     * Length of the encoding, in bytes, when using {@link #encode()} or {@link #encode(MutableDirectBuffer)}.
     * Length may be different using {@link #encode(OutputStream)}, due to things like stream headers
     * @return
     */
    int encodingLengthBytes();

    byte[] encode();

    void encode(MutableDirectBuffer buffer);

    void encode(OutputStream outputStream) throws IOException;

    void decode(byte[] decode);

    void decode(DirectBuffer buffer);

    void decode(InputStream inputStream) throws IOException;

}
