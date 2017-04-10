package org.deeplearning4j.spark.iterator;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * A DataSetIterator that loads serialized DataSet objects (saved with {@link DataSet#save(OutputStream)}) from
 * a {@link PortableDataStream}, usually obtained from SparkContext.binaryFiles()
 *
 * @author Alex Black
 */
public class PortableDataStreamDataSetIterator extends BaseDataSetIterator<PortableDataStream> {

    public PortableDataStreamDataSetIterator(Iterator<PortableDataStream> iter) {
        this.dataSetStreams = null;
        this.iter = iter;
    }

    public PortableDataStreamDataSetIterator(Collection<PortableDataStream> dataSetStreams) {
        this.dataSetStreams = dataSetStreams;
        iter = dataSetStreams.iterator();
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Total examples unknown for PortableDataStreamDataSetIterator");
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

        totalOutcomes = ds.getLabels().size(1);
        inputColumns = ds.getFeatureMatrix().size(1);
        batch = ds.numExamples();

        if (preprocessor != null)
            preprocessor.preProcess(ds);
        return ds;
    }

    protected DataSet load(PortableDataStream pds) {
        DataSet ds = new DataSet();
        try (InputStream is = pds.open()) {
            ds.load(is);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        cursor++;
        return ds;
    }

}
