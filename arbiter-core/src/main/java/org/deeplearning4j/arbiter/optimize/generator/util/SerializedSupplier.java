package org.deeplearning4j.arbiter.optimize.generator.util;

import org.nd4j.linalg.function.Supplier;

import java.io.*;

public class SerializedSupplier<T> implements Serializable, Supplier<T> {

    private byte[] asBytes;

    public SerializedSupplier(T obj){
        try(ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(obj);
            oos.flush();
            oos.close();
            asBytes = baos.toByteArray();
        } catch (Exception e){
            throw new RuntimeException("Error serializing object - must be serializable",e);
        }
    }

    @Override
    public T get() {
        try(ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(asBytes))){
            return (T)ois.readObject();
        } catch (Exception e){
            throw new RuntimeException("Error deserializing object",e);
        }
    }
}
