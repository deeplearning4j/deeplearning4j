package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

import static org.junit.Assert.assertEquals;

public class DataTypeTest {

    @Test
    public void testUINT16() throws IOException {
            INDArray in1 = Nd4j.ones(DataType.UINT16);
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test1.bin"));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test1.bin"));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch (ClassNotFoundException e) {

            }

            assertEquals(in1, in2);
    }

    @Test
    public void testUINT32() throws IOException {

            INDArray in1 = Nd4j.ones(DataType.UINT32);
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test2.bin"));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test2.bin"));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch (ClassNotFoundException e) {

            }

            assertEquals(in1, in2);
    }

    @Test
    public void testUINT64() throws IOException {

            INDArray in1 = Nd4j.ones(DataType.UINT64);
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test3.bin"));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test3.bin"));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch(ClassNotFoundException e) {

            }

            assertEquals(in1, in2);
    }

    @Test
    public void testInt() throws IOException {
        INDArray in1 = Nd4j.ones(DataType.INT);
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test1.bin"));
        oos.writeObject(in1);

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test1.bin"));
        INDArray in2 = null;
        try {
            in2 = (INDArray) ois.readObject();
        } catch (ClassNotFoundException e) {

        }

        assertEquals(in1, in2);
    }

    @Test
    public void testBfloat16() throws IOException {
        INDArray in1 = Nd4j.ones(DataType.BFLOAT16);
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test1.bin"));
        oos.writeObject(in1);

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test1.bin"));
        INDArray in2 = null;
        try {
            in2 = (INDArray) ois.readObject();
        } catch (ClassNotFoundException e) {

        }

        assertEquals(in1, in2);
    }

    @Test
    public void testFloat() throws IOException {
        INDArray in1 = Nd4j.ones(DataType.FLOAT);
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test1.bin"));
        oos.writeObject(in1);

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test1.bin"));
        INDArray in2 = null;
        try {
            in2 = (INDArray) ois.readObject();
        } catch (ClassNotFoundException e) {

        }

        assertEquals(in1, in2);
    }

    @Test
    public void testDouble() throws IOException {
        for (int i = 0; i < 10; ++i) {
            INDArray in1 = Nd4j.rand(DataType.DOUBLE, new int[]{100, 100});
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("test.bin"));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("test.bin"));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch(ClassNotFoundException e) {

            }

            assertEquals(in1, in2);
        }

    }
}
