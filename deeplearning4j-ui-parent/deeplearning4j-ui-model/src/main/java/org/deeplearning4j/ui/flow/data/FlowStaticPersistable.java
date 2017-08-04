package org.deeplearning4j.ui.flow.data;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.ui.flow.beans.ModelInfo;

import java.io.*;
import java.nio.ByteBuffer;

/**
 * Created by Alex on 25/10/2016.
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class FlowStaticPersistable implements Persistable {

    private String sessionID;
    private String workerID;
    private long timestamp;
    private ModelInfo modelInfo;

    @Override
    public String getSessionID() {
        return sessionID;
    }

    @Override
    public String getTypeID() {
        return FlowUpdatePersistable.TYPE_ID;
    }

    @Override
    public String getWorkerID() {
        return workerID;
    }

    @Override
    public long getTimeStamp() {
        return timestamp;
    }

    @Override
    public int encodingLengthBytes() {
        return 0;
    }

    @Override
    public byte[] encode() {
        //Not the most efficient: but it's easy to implement...
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e); //Shouldn't normally happen
        }

        return baos.toByteArray();
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        outputStream.write(encode());
    }

    @Override
    public void decode(byte[] decode) {
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decode))) {
            FlowStaticPersistable p = (FlowStaticPersistable) ois.readObject();
            this.sessionID = p.sessionID;
            this.workerID = p.workerID;
            this.timestamp = p.getTimeStamp();
            this.modelInfo = p.modelInfo;
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e); //Shouldn't normally happen
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        decode(arr);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        byte[] b = IOUtils.toByteArray(inputStream);
        decode(b);
    }
}
