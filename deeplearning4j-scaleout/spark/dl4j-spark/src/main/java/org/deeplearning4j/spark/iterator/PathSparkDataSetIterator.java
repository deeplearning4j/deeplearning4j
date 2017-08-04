package org.deeplearning4j.spark.iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * A DataSetIterator that loads serialized DataSet objects (saved with {@link DataSet#save(OutputStream)}) from
 * a String that represents the path (for example, on HDFS)
 *
 * @author Alex Black
 */
public class PathSparkDataSetIterator extends BaseDataSetIterator<String> {

    public static final int BUFFER_SIZE = 4194304; //4 MB
    private FileSystem fileSystem;

    public PathSparkDataSetIterator(Iterator<String> iter) {
        this.dataSetStreams = null;
        this.iter = iter;
    }

    public PathSparkDataSetIterator(Collection<String> dataSetStreams) {
        this.dataSetStreams = dataSetStreams;
        iter = dataSetStreams.iterator();
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Total examples unknown for PathSparkDataSetIterator");
    }

    @Override
    public DataSet next() {
        DataSet ds;
        if (preloadedDataSet != null) {
            ds = preloadedDataSet;
            preloadedDataSet = null;
        } else {
            ds = load(iter.next());
        }

        totalOutcomes = ds.getLabels() == null ? 0 : ds.getLabels().size(1); //May be null for layerwise pretraining
        inputColumns = ds.getFeatureMatrix().size(1);
        batch = ds.numExamples();

        if (preprocessor != null)
            preprocessor.preProcess(ds);
        return ds;
    }

    protected synchronized DataSet load(String path) {
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

        cursor++;
        return ds;
    }
}
