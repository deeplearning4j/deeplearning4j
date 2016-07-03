package org.deeplearning4j.spark.util;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.writable.Writable;

import java.io.*;
import java.util.List;

/**
 * Various utilities for Spark
 *
 * @author Alex Black
 */
public class SparkUtils {

    private SparkUtils() {
    }

    /**
     * Write a String to a file (on HDFS or local) in UTF-8 format
     *
     * @param path       Path to write to
     * @param toWrite    String to write
     * @param sc         Spark context
     */
    public static void writeStringToFile(String path, String toWrite, JavaSparkContext sc) throws IOException {
        writeStringToFile(path, toWrite, sc.sc());
    }

    /**
     * Write a String to a file (on HDFS or local) in UTF-8 format
     *
     * @param path       Path to write to
     * @param toWrite    String to write
     * @param sc         Spark context
     */
    public static void writeStringToFile(String path, String toWrite, SparkContext sc) throws IOException {
        FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
        try(BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(path)))){
            bos.write(toWrite.getBytes("UTF-8"));
        }
    }

    /**
     * Read a UTF-8 format String from HDFS (or local)
     *
     * @param path    Path to write the string
     * @param sc      Spark context
     */
    public static String readStringFromFile(String path, JavaSparkContext sc) throws IOException {
        return readStringFromFile(path, sc.sc());
    }

    /**
     * Read a UTF-8 format String from HDFS (or local)
     *
     * @param path    Path to write the string
     * @param sc      Spark context
     */
    public static String readStringFromFile(String path, SparkContext sc) throws IOException {
        FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
        try(BufferedInputStream bis = new BufferedInputStream(fileSystem.open(new Path(path)))){
            byte[] asBytes = IOUtils.toByteArray(bis);
            return new String(asBytes,"UTF-8");
        }
    }

    /**
     * Write an object to HDFS (or local) using default Java object serialization
     *
     * @param path       Path to write the object to
     * @param toWrite    Object to write
     * @param sc         Spark context
     */
    public static void writeObjectToFile(String path, Object toWrite, JavaSparkContext sc) throws IOException {
        writeObjectToFile(path, toWrite, sc.sc());
    }

    /**
     * Write an object to HDFS (or local) using default Java object serialization
     *
     * @param path       Path to write the object to
     * @param toWrite    Object to write
     * @param sc         Spark context
     */
    public static void writeObjectToFile(String path, Object toWrite, SparkContext sc) throws IOException {
        FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
        try(BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(path)))){
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(toWrite);
        }
    }

    /**
     * Read an object from HDFS (or local) using default Java object serialization
     *
     * @param path    File to read
     * @param type    Class of the object to read
     * @param sc      Spark context
     * @param <T>     Type of the object to read
     */
    public static <T> T readObjectFromFile(String path, Class<T> type, JavaSparkContext sc) throws IOException {
        return readObjectFromFile(path, type, sc.sc());
    }

    /**
     * Read an object from HDFS (or local) using default Java object serialization
     *
     * @param path    File to read
     * @param type    Class of the object to read
     * @param sc      Spark context
     * @param <T>     Type of the object to read
     */
    public static <T> T readObjectFromFile(String path, Class<T> type, SparkContext sc) throws IOException {
        FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
        try(ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(fileSystem.open(new Path(path))))){
            Object o;
            try {
                o = ois.readObject();
            } catch( ClassNotFoundException e ){
                throw new RuntimeException(e);
            }

            return (T)o;
        }
    }
}
