package org.nd4j.linalg.api.ndarray;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 28/04/2016.
 */
@RunWith(Parameterized.class)
public class TestSerialization extends BaseNd4jTest {

    public TestSerialization(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testSerializationFullArrayNd4jWriteRead() throws Exception {
        int length = 100;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',10,10);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try(DataOutputStream dos = new DataOutputStream(baos)){
            Nd4j.write(arr,dos);
        }
        byte[] bytes = baos.toByteArray();

        INDArray arr2;
        try( DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))){
            arr2 = Nd4j.read(dis);
        }

        assertEquals(arr,arr2);
    }

    @Test
    public void testSerializationFullArrayJava() throws Exception {
        int length = 100;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',10,10);



        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try(ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(arr);
        }
        byte[] bytes = baos.toByteArray();

        INDArray arr2;
        try( ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))){
            arr2 = (INDArray) ois.readObject();
        }

        assertEquals(arr,arr2);
    }

    @Test
    public void testSerializationOnViewsNd4jWriteRead() throws Exception {
        int length = 100;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',10,10);

        INDArray sub = arr.get(NDArrayIndex.interval(5,10), NDArrayIndex.interval(5,10));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try(DataOutputStream dos = new DataOutputStream(baos)){
            Nd4j.write(sub,dos);
        }
        byte[] bytes = baos.toByteArray();

        INDArray arr2;
        try( DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))){
            arr2 = Nd4j.read(dis);
        }

        assertEquals(sub,arr2);
    }

    @Test
    public void testSerializationOnViewsJava() throws Exception {
        int length = 100;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',10,10);

        INDArray sub = arr.get(NDArrayIndex.interval(5,10), NDArrayIndex.interval(5,10));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try(ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(sub);
        }
        byte[] bytes = baos.toByteArray();

        INDArray arr2;
        try( ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))){
            arr2 = (INDArray) ois.readObject();
        }

        assertEquals(sub,arr2);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
