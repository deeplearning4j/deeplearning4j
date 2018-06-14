package org.deeplearning4j.datasets.datavec;

import lombok.Getter;
import lombok.Setter;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataComposable;
import org.datavec.api.records.metadata.RecordMetaDataComposableMap;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.deeplearning4j.datasets.datavec.exception.ZeroLengthSequenceException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Sequence record reader data set iterator
 * Given a record reader (and optionally another record reader for the labels) generate time series (sequence) data sets.<br>
 * Supports padding for one-to-many and many-to-one type data loading (i.e., with different number of inputs vs.
 * labels via the {@link AlignmentMode} mode.
 *
 * @author Alex Black
 */
public class SequenceRecordReaderDataSetIterator implements DataSetIterator {
    /**Alignment mode for dealing with input/labels of differing lengths (for example, one-to-many and many-to-one type situations).
     * For example, might have 10 time steps total but only one label at end for sequence classification.<br>
     * Currently supported modes:<br>
     * <b>EQUAL_LENGTH</b>: Default. Assume that label and input time series are of equal length, and all examples are of
     * the same length<br>
     * <b>ALIGN_START</b>: Align the label/input time series at the first time step, and zero pad either the labels or
     * the input at the end<br>
     * <b>ALIGN_END</b>: Align the label/input at the last time step, zero padding either the input or the labels as required<br>
     *
     * Note 1: When the time series for each example are of different lengths, the shorter time series will be padded to
     * the length of the longest time series.<br>
     * Note 2: When ALIGN_START or ALIGN_END are used, the DataSet masking functionality is used. Thus, the returned DataSets
     * will have the input and mask arrays set. These mask arrays identify whether an input/label is actually present,
     * or whether the value is merely masked.<br>
     */
    public enum AlignmentMode {
        EQUAL_LENGTH, ALIGN_START, ALIGN_END
    }

    private static final String READER_KEY = "reader";
    private static final String READER_KEY_LABEL = "reader_labels";

    private SequenceRecordReader recordReader;
    private SequenceRecordReader labelsReader;
    private int miniBatchSize = 10;
    private final boolean regression;
    private int labelIndex = -1;
    private final int numPossibleLabels;
    private int cursor = 0;
    private int inputColumns = -1;
    private int totalOutcomes = -1;
    private boolean useStored = false;
    private DataSet stored = null;
    @Getter
    private DataSetPreProcessor preProcessor;
    private AlignmentMode alignmentMode;

    private final boolean singleSequenceReaderMode;

    @Getter
    @Setter
    private boolean collectMetaData = false;

    private RecordReaderMultiDataSetIterator underlying;
    private boolean underlyingIsDisjoint;

    /**
     * Constructor where features and labels come from different RecordReaders (for example, different files),
     * and labels are for classification.
     *
     * @param featuresReader       SequenceRecordReader for the features
     * @param labels               Labels: assume single value per time step, where values are integers in the range 0 to numPossibleLables-1
     * @param miniBatchSize        Minibatch size for each call of next()
     * @param numPossibleLabels    Number of classes for the labels
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader featuresReader, SequenceRecordReader labels,
                    int miniBatchSize, int numPossibleLabels) {
        this(featuresReader, labels, miniBatchSize, numPossibleLabels, false);
    }

    /**
     * Constructor where features and labels come from different RecordReaders (for example, different files)
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader featuresReader, SequenceRecordReader labels,
                    int miniBatchSize, int numPossibleLabels, boolean regression) {
        this(featuresReader, labels, miniBatchSize, numPossibleLabels, regression, AlignmentMode.EQUAL_LENGTH);
    }

    /**
     * Constructor where features and labels come from different RecordReaders (for example, different files)
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader featuresReader, SequenceRecordReader labels,
                    int miniBatchSize, int numPossibleLabels, boolean regression, AlignmentMode alignmentMode) {
        this.recordReader = featuresReader;
        this.labelsReader = labels;
        this.miniBatchSize = miniBatchSize;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
        this.alignmentMode = alignmentMode;
        this.singleSequenceReaderMode = false;
    }

    /** Constructor where features and labels come from the SAME RecordReader (i.e., target/label is a column in the
     * same data as the features). Defaults to regression = false - i.e., for classification
     * @param reader SequenceRecordReader with data
     * @param miniBatchSize size of each minibatch
     * @param numPossibleLabels number of labels/classes for classification
     * @param labelIndex index in input of the label index. If in regression mode and numPossibleLabels > 1, labelIndex denotes the
     *                   first index for labels. Everything before that index will be treated as input(s) and
     *                   everything from that index (inclusive) to the end will be treated as output(s)
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader reader, int miniBatchSize, int numPossibleLabels,
                    int labelIndex) {
        this(reader, miniBatchSize, numPossibleLabels, labelIndex, false);
    }

    /** Constructor where features and labels come from the SAME RecordReader (i.e., target/label is a column in the
     * same data as the features)
     * @param reader SequenceRecordReader with data
     * @param miniBatchSize size of each minibatch
     * @param numPossibleLabels number of labels/classes for classification
     * @param labelIndex index in input of the label index. If in regression mode and numPossibleLabels > 1, labelIndex denotes the
     *                   first index for labels. Everything before that index will be treated as input(s) and
     *                   everything from that index (inclusive) to the end will be treated as output(s)
     * @param regression Whether output is for regression or classification
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader reader, int miniBatchSize, int numPossibleLabels,
                    int labelIndex, boolean regression) {
        this.recordReader = reader;
        this.labelsReader = null;
        this.miniBatchSize = miniBatchSize;
        this.regression = regression;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.singleSequenceReaderMode = true;
    }

    private void initializeUnderlyingFromReader() {
        initializeUnderlying(recordReader.nextSequence());
        underlying.reset();
    }

    private void initializeUnderlying(SequenceRecord nextF) {
        if (nextF.getSequenceRecord().isEmpty()) {
            throw new ZeroLengthSequenceException();
        }
        int totalSizeF = nextF.getSequenceRecord().get(0).size();

        //allow people to specify label index as -1 and infer the last possible label
        if (singleSequenceReaderMode && numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = totalSizeF - 1;
        } else if (!singleSequenceReaderMode && numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = 0;
        }

        recordReader.reset();

        //Add readers
        RecordReaderMultiDataSetIterator.Builder builder = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize);
        builder.addSequenceReader(READER_KEY, recordReader);
        if (labelsReader != null) {
            builder.addSequenceReader(READER_KEY_LABEL, labelsReader);
        }


        //Add outputs
        if (singleSequenceReaderMode) {
            //Features: subset of columns
            if (labelIndex == 0 || labelIndex == totalSizeF - 1) {
                //Labels are first or last -> one input in underlying
                int inputFrom;
                int inputTo;
                if (labelIndex < 0) {
                    //No label
                    inputFrom = 0;
                    inputTo = totalSizeF - 1;
                } else if (labelIndex == 0) {
                    inputFrom = 1;
                    inputTo = totalSizeF - 1;
                } else {
                    inputFrom = 0;
                    inputTo = labelIndex - 1;
                }

                builder.addInput(READER_KEY, inputFrom, inputTo);

                underlyingIsDisjoint = false;
            } else if (regression && numPossibleLabels > 1){
                //Multiple inputs and multiple outputs
                int inputFrom = 0;
                int inputTo = labelIndex - 1;
                int outputFrom = labelIndex;
                int outputTo = totalSizeF - 1;

                builder.addInput(READER_KEY, inputFrom, inputTo);
                builder.addOutput(READER_KEY, outputFrom, outputTo);

                underlyingIsDisjoint = false;
            } else {
                //Multiple inputs (disjoint features case)
                int firstFrom = 0;
                int firstTo = labelIndex - 1;
                int secondFrom = labelIndex + 1;
                int secondTo = totalSizeF - 1;

                builder.addInput(READER_KEY, firstFrom, firstTo);
                builder.addInput(READER_KEY, secondFrom, secondTo);

                underlyingIsDisjoint = true;
            }


            //Multiple output regression already handled
            if (regression && numPossibleLabels <= 1) {
                builder.addOutput(READER_KEY, labelIndex, labelIndex);
            } else if (!regression) {
                builder.addOutputOneHot(READER_KEY, labelIndex, numPossibleLabels);
            }
        } else {

            //Features: entire reader
            builder.addInput(READER_KEY);
            underlyingIsDisjoint = false;

            if (regression) {
                builder.addOutput(READER_KEY_LABEL);
            } else {
                builder.addOutputOneHot(READER_KEY_LABEL, 0, numPossibleLabels);
            }
        }

        if (alignmentMode != null) {
            switch (alignmentMode) {
                case EQUAL_LENGTH:
                    builder.sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.EQUAL_LENGTH);
                    break;
                case ALIGN_START:
                    builder.sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START);
                    break;
                case ALIGN_END:
                    builder.sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END);
                    break;
            }
        }

        underlying = builder.build();

        if (collectMetaData) {
            underlying.setCollectMetaData(true);
        }
    }

    private DataSet mdsToDataSet(MultiDataSet mds) {
        INDArray f;
        INDArray fm;
        if (underlyingIsDisjoint) {
            //Rare case: 2 input arrays -> concat
            INDArray f1 = RecordReaderDataSetIterator.getOrNull(mds.getFeatures(), 0);
            INDArray f2 = RecordReaderDataSetIterator.getOrNull(mds.getFeatures(), 1);
            fm = RecordReaderDataSetIterator.getOrNull(mds.getFeaturesMaskArrays(), 0); //Per-example masking only on the input -> same for both

            //Can assume 3d features here
            f = Nd4j.createUninitialized(new long[] {f1.size(0), f1.size(1) + f2.size(1), f1.size(2)});
            f.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(0, f1.size(1)), NDArrayIndex.all()},
                            f1);
            f.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(f1.size(1), f1.size(1) + f2.size(1)),
                            NDArrayIndex.all()}, f2);
        } else {
            //Standard case
            f = RecordReaderDataSetIterator.getOrNull(mds.getFeatures(), 0);
            fm = RecordReaderDataSetIterator.getOrNull(mds.getFeaturesMaskArrays(), 0);
        }

        INDArray l = RecordReaderDataSetIterator.getOrNull(mds.getLabels(), 0);
        INDArray lm = RecordReaderDataSetIterator.getOrNull(mds.getLabelsMaskArrays(), 0);

        DataSet ds = new DataSet(f, l, fm, lm);

        if (collectMetaData) {
            List<Serializable> temp = mds.getExampleMetaData();
            List<Serializable> temp2 = new ArrayList<>(temp.size());
            for (Serializable s : temp) {
                RecordMetaDataComposableMap m = (RecordMetaDataComposableMap) s;
                if (singleSequenceReaderMode) {
                    temp2.add(m.getMeta().get(READER_KEY));
                } else {
                    RecordMetaDataComposable c = new RecordMetaDataComposable(m.getMeta().get(READER_KEY),
                                    m.getMeta().get(READER_KEY_LABEL));
                    temp2.add(c);
                }
            }
            ds.setExampleMetaData(temp2);
        }

        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }

        return ds;
    }

    @Override
    public boolean hasNext() {
        if (underlying == null) {
            initializeUnderlyingFromReader();
        }
        return underlying.hasNext();
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }


    @Override
    public DataSet next(int num) {
        if (useStored) {
            useStored = false;
            DataSet temp = stored;
            stored = null;
            if (preProcessor != null)
                preProcessor.preProcess(temp);
            return temp;
        }
        if (!hasNext())
            throw new NoSuchElementException();

        if (underlying == null) {
            initializeUnderlyingFromReader();
        }

        MultiDataSet mds = underlying.next(num);
        DataSet ds = mdsToDataSet(mds);

        if (totalOutcomes == -1) {
            // FIXME: int cast
            inputColumns = (int) ds.getFeatures().size(1);
            totalOutcomes = (int) ds.getLabels().size(1);
        }

        return ds;
    }

    @Override
    public int inputColumns() {
        if (inputColumns != -1)
            return inputColumns;
        preLoad();
        return inputColumns;
    }

    @Override
    public int totalOutcomes() {
        if (totalOutcomes != -1)
            return totalOutcomes;
        preLoad();
        return totalOutcomes;
    }

    private void preLoad() {
        stored = next();
        useStored = true;

        // FIXME: int cast
        inputColumns = (int) stored.getFeatureMatrix().size(1);
        totalOutcomes = (int) stored.getLabels().size(1);
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        if (underlying != null)
            underlying.reset();

        cursor = 0;
        stored = null;
        useStored = false;
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove not supported for this iterator");
    }

    /**
     * Load a single sequence example to a DataSet, using the provided RecordMetaData.
     * Note that it is more efficient to load multiple instances at once, using {@link #loadFromMetaData(List)}
     *
     * @param recordMetaData RecordMetaData to load from. Should have been produced by the given record reader
     * @return DataSet with the specified example
     * @throws IOException If an error occurs during loading of the data
     */
    public DataSet loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData));
    }

    /**
     * Load a multiple sequence examples to a DataSet, using the provided RecordMetaData instances.
     *
     * @param list List of RecordMetaData instances to load from. Should have been produced by the record reader provided
     *             to the SequenceRecordReaderDataSetIterator constructor
     * @return DataSet with the specified examples
     * @throws IOException If an error occurs during loading of the data
     */
    public DataSet loadFromMetaData(List<RecordMetaData> list) throws IOException {
        if (underlying == null) {
            SequenceRecord r = recordReader.loadSequenceFromMetaData(list.get(0));
            initializeUnderlying(r);
        }

        //Two cases: single vs. multiple reader...
        List<RecordMetaData> l = new ArrayList<>(list.size());
        if (singleSequenceReaderMode) {
            for (RecordMetaData m : list) {
                l.add(new RecordMetaDataComposableMap(Collections.singletonMap(READER_KEY, m)));
            }
        } else {
            for (RecordMetaData m : list) {
                RecordMetaDataComposable rmdc = (RecordMetaDataComposable) m;
                Map<String, RecordMetaData> map = new HashMap<>(2);
                map.put(READER_KEY, rmdc.getMeta()[0]);
                map.put(READER_KEY_LABEL, rmdc.getMeta()[1]);
                l.add(new RecordMetaDataComposableMap(map));
            }
        }

        return mdsToDataSet(underlying.loadFromMetaData(l));
    }
}
