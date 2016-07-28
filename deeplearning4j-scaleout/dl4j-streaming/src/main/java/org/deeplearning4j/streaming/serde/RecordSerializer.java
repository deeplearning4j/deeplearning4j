package org.deeplearning4j.streaming.serde;

import org.apache.commons.io.IOUtils;
import org.apache.kafka.common.serialization.Serializer;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * Serializes records for kafka
 *
 * @author Adam Gibson
 *
 * */
public class RecordSerializer implements Serializer<Collection<Collection<Writable>>> {
    @Override
    public void configure(Map<String, ?> map, boolean b) {

    }

    @Override
    public byte[] serialize(String s, Collection<Collection<Writable>> writables) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        DataOutputStream dataOutputStream = new DataOutputStream(byteArrayOutputStream);
        SerializationUtils.writeObject((Serializable) writables,dataOutputStream);
        try {
            byteArrayOutputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            IOUtils.closeQuietly(byteArrayOutputStream);
        }

        byte[] ret = byteArrayOutputStream.toByteArray();
        return ret;
    }

    @Override
    public void close() {

    }
}
