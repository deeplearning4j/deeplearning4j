package org.deeplearning4j.spark.data;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.VoidFunction;
import org.canova.api.io.converters.SelfWritableConverter;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CollectionRecordReader;
import org.canova.api.split.StringSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.spark.util.UIDProvider;
import org.nd4j.linalg.dataset.DataSet;

import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

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
        uid = Thread.currentThread().getId() + jvmuid.substring(0,Math.min(8,jvmuid.length()));


        while(iter.hasNext()){
            DataSet next = iter.next();

            String filename = "dataset_" + uid + "_" + (outputCount++) + ".bin";

            URI uri = new URI(outputDir.getPath() + "/" + filename);
            FileSystem file = FileSystem.get(uri, conf);
            try(FSDataOutputStream out = file.create(new Path(uri))){
                next.save(out);
            }
        }
    }
}
