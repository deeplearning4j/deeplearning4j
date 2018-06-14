package org.deeplearning4j.spark.iterator;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Iterator;

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
    public DataSet next() {
        DataSet ds;
        if (preloadedDataSet != null) {
            ds = preloadedDataSet;
            preloadedDataSet = null;
        } else {
            ds = load(iter.next());
        }

        // FIXME: int cast
        totalOutcomes = (int) ds.getLabels().size(1);
        inputColumns = (int) ds.getFeatureMatrix().size(1);
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
            throw new RuntimeException("Error loading DataSet at path " + pds.getPath() + " - DataSet may be corrupt or invalid." +
                    " Spark DataSets can be validated using org.deeplearning4j.spark.util.data.SparkDataValidation", e);
        }
        cursor++;
        return ds;
    }

}
