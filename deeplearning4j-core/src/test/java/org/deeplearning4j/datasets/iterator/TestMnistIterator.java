package org.deeplearning4j.datasets.iterator;

import org.canova.api.io.WritableConverter;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;


/**
 * Created by agibsonccc on 3/8/16.
 */
public class TestMnistIterator extends RecordReaderDataSetIterator {
    public TestMnistIterator() {
        this(getTestMnistRecordReader(),10,0,10);

    }

    public static RecordReader getTestMnistRecordReader() {
        RecordReader csv = new CSVRecordReader();
        InputSplit file;
        try {
            file = new FileSplit(new ClassPathResource("mnist_first_200.txt").getFile());
            csv.initialize(file);
            return csv;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    public TestMnistIterator(RecordReader recordReader, int batchSize) {
        super(recordReader, batchSize);
    }

    public TestMnistIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels) {
        super(recordReader, batchSize, labelIndex, numPossibleLabels);
    }

    public TestMnistIterator(RecordReader recordReader) {
        super(recordReader);
    }

    public TestMnistIterator(RecordReader recordReader, int labelIndex, int numPossibleLabels) {
        super(recordReader, labelIndex, numPossibleLabels);
    }

    public TestMnistIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels, boolean regression) {
        super(recordReader, converter, batchSize, labelIndex, numPossibleLabels, regression);
    }

    public TestMnistIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels) {
        super(recordReader, converter, batchSize, labelIndex, numPossibleLabels);
    }

    public TestMnistIterator(RecordReader recordReader, WritableConverter converter) {
        super(recordReader, converter);
    }

    public TestMnistIterator(RecordReader recordReader, WritableConverter converter, int labelIndex, int numPossibleLabels) {
        super(recordReader, converter, labelIndex, numPossibleLabels);
    }
}
