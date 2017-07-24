package org.deeplearning4j.arbiter.ui.data;

import lombok.AllArgsConstructor;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.arbiter.ui.module.ArbiterModule;
import org.deeplearning4j.ui.stats.impl.java.JavaStatsInitializationReport;
import scala.annotation.meta.field;

import java.io.*;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Common implementation
 *
 * @author Alex Black
 */
@AllArgsConstructor
public abstract class BaseJavaPersistable implements Persistable {

    private String sessionId;
    private long timestamp;

    public BaseJavaPersistable(Builder builder){
        this.sessionId = builder.sessionId;
        this.timestamp = builder.timestamp;
    }

    protected BaseJavaPersistable(){
        //No-arg costructor for Pesistable encoding/decoding
    }

    @Override
    public String getTypeID() {
        return ArbiterModule.ARBITER_UI_TYPE_ID;
    }

    @Override
    public long getTimeStamp() {
        return timestamp;
    }

    @Override
    public String getSessionID() {
        return sessionId;
    }

    @Override
    public int encodingLengthBytes() {
        //TODO - presumably a more efficient way to do this
        byte[] encoded = encode();
        return encoded.length;
    }

    @Override
    public byte[] encode() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e); //Should never happen
        }
        return baos.toByteArray();
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(outputStream)) {
            oos.writeObject(this);
        }
    }

    @Override
    public void decode(byte[] decode) {
        BaseJavaPersistable r;
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decode))) {
            r = (BaseJavaPersistable) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e); //Should never happen
        }

        //Need to manually build and walk the class heirarchy...
        Class<?> currClass = this.getClass();
        List<Class<?>> classHeirarchy = new ArrayList<>();
        while (currClass != Object.class) {
            classHeirarchy.add(currClass);
            currClass = currClass.getSuperclass();
        }

        for (int i = classHeirarchy.size() - 1; i >= 0; i--) {
            //Use reflection here to avoid a mass of boilerplate code...
            Field[] allFields = classHeirarchy.get(i).getDeclaredFields();

            for (Field f : allFields) {
                if (Modifier.isStatic(f.getModifiers())) {
                    //Skip static fields
                    continue;
                }
                f.setAccessible(true);
                try {
                    f.set(this, f.get(r));
                } catch (IllegalAccessException e) {
                    throw new RuntimeException(e); //Should never happen
                }
            }
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        decode(bytes);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        decode(IOUtils.toByteArray(inputStream));
    }

    public static abstract class Builder<T extends Builder<T>> {
        protected String sessionId;
        protected long timestamp;

        public T sessionId(String sessionId){
            this.sessionId = sessionId;
            return (T) this;
        }

        public T timestamp(long timestamp){
            this.timestamp = timestamp;
            return (T) this;
        }

    }
}
