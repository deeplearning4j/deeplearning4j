package org.deeplearning4j.streaming.serde;

import kafka.serializer.Decoder;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;

/**
 * Created by agibsonccc on 6/7/16.
 */
public class RecordDecoder implements Decoder<Object> {
    @Override
    public Object fromBytes(byte[] bytes) {
        ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
        BufferedInputStream bis2 = new BufferedInputStream(bis);
        return SerializationUtils.readObject(bis2);
    }
}
