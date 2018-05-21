package org.deeplearning4j.spark.models.sequencevectors.export.impl;

import lombok.NonNull;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;

/**
 * Simple exporter, that will persist your SequenceVectors model into HDFS using TSV format
 *
 * @author raver119@gmail.com
 */
public class HdfsModelExporter<T extends SequenceElement> implements SparkModelExporter<T> {
    protected String path;
    protected CompressionCodec codec;

    protected HdfsModelExporter() {

    }

    public HdfsModelExporter(@NonNull String path) {
        this(path, null);
    }

    public HdfsModelExporter(@NonNull String path, CompressionCodec codec) {
        this.path = path;
        this.codec = codec;
    }

    @Override
    public void export(JavaRDD<ExportContainer<T>> rdd) {
        if (codec == null)
            rdd.saveAsTextFile(path);
        else
            rdd.saveAsTextFile(path, codec.getClass());
    }
}
