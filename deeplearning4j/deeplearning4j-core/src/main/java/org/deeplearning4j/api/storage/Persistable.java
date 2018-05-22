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

    /**
     * Get the session id
     * @return
     */
    String getSessionID();

    /**
     * Get the type id
     * @return
     */
    String getTypeID();

    /**
     * Get the worker id
     * @return
     */
    String getWorkerID();

    /**
     * Get when this was created.
     * @return
     */
    long getTimeStamp();


    //SerDe methods:

    /**
     * Length of the encoding, in bytes, when using {@link #encode()}
     * Length may be different using {@link #encode(OutputStream)}, due to things like stream headers
     * @return
     */
    int encodingLengthBytes();

    byte[] encode();

    /**
     * Encode this persistable in to a {@link ByteBuffer}
     * @param buffer
     */
    void encode(ByteBuffer buffer);

    /**
     * Encode this persistable in to an output stream
     * @param outputStream
     * @throws IOException
     */
    void encode(OutputStream outputStream) throws IOException;

    /**
     * Decode the content of the given
     * byte array in to this persistable
     * @param decode
     */
    void decode(byte[] decode);

    /**
     * Decode from the given {@link ByteBuffer}
     * @param buffer
     */
    void decode(ByteBuffer buffer);

    /**
     * Decode from the given input stream
     * @param inputStream
     * @throws IOException
     */
    void decode(InputStream inputStream) throws IOException;

}
