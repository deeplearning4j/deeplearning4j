package org.nd4j.linalg.api.buffer.factory;

/**
 * Created by agibsonccc on 2/25/16.
 */
public class AllocatorInstanceHolder {
    private static DataBufferFactory INSTANCE;

    public static DataBufferFactory getInstance() {
        if(INSTANCE == null)
            INSTANCE = new DefaultDataBufferFactory();
        return INSTANCE;
    }

}
