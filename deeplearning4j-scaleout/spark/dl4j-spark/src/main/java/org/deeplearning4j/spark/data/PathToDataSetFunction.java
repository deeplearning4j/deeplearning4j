package org.deeplearning4j.spark.data;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.net.URI;

/**
 * Simple function used to load DataSets (serialized with DataSet.save()) from a given Path (as a String)
 * to a DataSet object - i.e., {@code RDD<String>} to {@code RDD<DataSet>}
 *
 * @author Alex Black
 */
public class PathToDataSetFunction implements Function<String, DataSet> {
    public static final int BUFFER_SIZE = 4194304; //4 MB

    private FileSystem fileSystem;

    @Override
    public DataSet call(String path) throws Exception {
        if (fileSystem == null) {
            try {
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        DataSet ds = new DataSet();
        try (FSDataInputStream inputStream = fileSystem.open(new Path(path), BUFFER_SIZE)) {
            ds.load(inputStream);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return ds;
    }
}
