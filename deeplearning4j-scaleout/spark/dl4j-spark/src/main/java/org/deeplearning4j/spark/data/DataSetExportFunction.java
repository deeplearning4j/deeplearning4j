package org.deeplearning4j.spark.data;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.dataset.DataSet;

import java.net.URI;
import java.util.Iterator;

/**
 * A function (used in forEachPartition) to save DataSet objects to disk/HDFS. Each DataSet object is given a random and
 * (probably) unique name, starting with "dataset_" and ending  with ".bin".<br>
 * Use with {@code JavaRDD<DataSet>.foreachPartition()}
 *
 * @author Alex Black
 */
public class DataSetExportFunction implements VoidFunction<Iterator<DataSet>> {
    private static final Configuration conf = new Configuration();

    private final URI outputDir;
    private String uid = null;

    private int outputCount;

    public DataSetExportFunction(URI outputDir) {
        this.outputDir = outputDir;
    }

    @Override
    public void call(Iterator<DataSet> iter) throws Exception {
        String jvmuid = UIDProvider.getJVMUID();
        uid = Thread.currentThread().getId() + jvmuid.substring(0, Math.min(8, jvmuid.length()));


        while (iter.hasNext()) {
            DataSet next = iter.next();

            String filename = "dataset_" + uid + "_" + (outputCount++) + ".bin";

            String path = outputDir.getPath();
            URI uri = new URI(path + (path.endsWith("/") || path.endsWith("\\") ? "" : "/") + filename);
            FileSystem file = FileSystem.get(uri, conf);
            try (FSDataOutputStream out = file.create(new Path(uri))) {
                next.save(out);
            }
        }
    }
}
