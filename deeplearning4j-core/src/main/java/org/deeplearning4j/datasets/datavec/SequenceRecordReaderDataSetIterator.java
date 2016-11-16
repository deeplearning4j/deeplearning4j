package org.deeplearning4j.datasets.datavec;

import lombok.Getter;
import lombok.Setter;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataComposable;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.deeplearning4j.datasets.datavec.exception.ZeroLengthSequenceException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.IOException;
import java.util.*;

/**
 * Sequence record reader data set iterator
 * Given a record reader (and optionally another record reader for the labels) generate time series (sequence) data sets.<br>
 * Supports padding for one-to-many and many-to-one type data loading (i.e., with different number of inputs vs.
 * labels via the {@link SequenceRecordReaderDataSetIterator.AlignmentMode} mode.
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
        EQUAL_LENGTH,
        ALIGN_START,
        ALIGN_END
    }
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
    @Getter private DataSetPreProcessor preProcessor;
    private AlignmentMode alignmentMode;

    private final boolean singleSequenceReaderMode;

    @Getter @Setter
    private boolean collectMetaData = false;

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
        this(featuresReader,labels,miniBatchSize,numPossibleLabels,regression, AlignmentMode.EQUAL_LENGTH);
    }

    /**
     * Constructor where features and labels come from different RecordReaders (for example, different files)
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader featuresReader, SequenceRecordReader labels,
                                               int miniBatchSize, int numPossibleLabels, boolean regression,
                                               AlignmentMode alignmentMode) {
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
     * @param numPossibleLabels number of labels/classes for classification (or not used if regression == true)
     * @param labelIndex index in input of the label index
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader reader, int miniBatchSize, int numPossibleLabels, int labelIndex){
        this(reader, miniBatchSize, numPossibleLabels, labelIndex, false);
    }

    /** Constructor where features and labels come from the SAME RecordReader (i.e., target/label is a column in the
     * same data as the features)
     * @param reader SequenceRecordReader with data
     * @param miniBatchSize size of each minibatch
     * @param numPossibleLabels number of labels/classes for classification (or not used if regression == true)
     * @param labelIndex index in input of the label index
     * @param regression Whether output is for regression or classification
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader reader, int miniBatchSize, int numPossibleLabels,
                                               int labelIndex, boolean regression){
        this.recordReader = reader;
        this.labelsReader = null;
        this.miniBatchSize = miniBatchSize;
        this.regression = regression;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.singleSequenceReaderMode = true;
    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
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
            if (preProcessor != null) preProcessor.preProcess(temp);
            return temp;
        }
        if (!hasNext()) throw new NoSuchElementException();

        if (singleSequenceReaderMode) {
            return nextSingleSequenceReader(num);
        } else {
            return nextMultipleSequenceReaders(num);
        }
    }

    private DataSet nextSingleSequenceReader(int num) {
        List<INDArray> listFeatures = new ArrayList<>(num);
        List<INDArray> listLabels = new ArrayList<>(num);
        List<RecordMetaData> meta = (collectMetaData ? new ArrayList<RecordMetaData>() : null);
        int minLength = 0;
        int maxLength = 0;
        for (int i = 0; i < num && hasNext(); i++) {
            List<List<Writable>> sequence;
            if (collectMetaData ) {
                SequenceRecord sequenceRecord = recordReader.nextSequence();
                sequence = sequenceRecord.getSequenceRecord();
                meta.add(sequenceRecord.getMetaData());
            } else {
                sequence = recordReader.sequenceRecord();
            }
            assertNonZeroLengthSequence(sequence, "combined features and labels");

            INDArray[] fl = getFeaturesLabelsSingleReader(sequence);
            if (i == 0) {
                minLength = fl[0].size(0);
                maxLength = minLength;
            } else {
                minLength = Math.min(minLength, fl[0].size(0));
                maxLength = Math.max(maxLength, fl[0].size(0));
            }
            listFeatures.add(fl[0]);
            listLabels.add(fl[1]);
        }

        return getSingleSequenceReader(listFeatures, listLabels, minLength, maxLength, meta);
    }

    private DataSet getSingleSequenceReader(List<INDArray> listFeatures, List<INDArray> listLabels, int minLength, int maxLength,
                                            List<RecordMetaData> meta ){

        //Convert to 3d minibatch
        //Note: using f order here, as each  time step is contiguous in the buffer with f order (isn't the case with c order)
        INDArray featuresOut = Nd4j.create(new int[]{listFeatures.size(),listFeatures.get(0).size(1),maxLength},'f');
        INDArray labelsOut = Nd4j.create(new int[]{listLabels.size(),listLabels.get(0).size(1),maxLength},'f');
        INDArray featuresMask = null;
        INDArray labelsMask = null;

        if(minLength == maxLength){
            for (int i = 0; i < listFeatures.size(); i++) {
                //Note: this TAD gives us shape [vectorSize,tsLength] whereas we need a [vectorSize,timeSeriesLength] matrix (that listFeatures contains)
                featuresOut.tensorAlongDimension(i, 1, 2).permutei(1,0).assign(listFeatures.get(i));
                labelsOut.tensorAlongDimension(i, 1, 2).permutei(1,0).assign(listLabels.get(i));
            }
        } else {
            featuresMask = Nd4j.ones(listFeatures.size(),maxLength);
            labelsMask = Nd4j.ones(listLabels.size(),maxLength);
            for (int i = 0; i < listFeatures.size(); i++) {
                INDArray f = listFeatures.get(i);
                int tsLength = f.size(0);

                featuresOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                        .put(new INDArrayIndex[]{NDArrayIndex.interval(0, tsLength), NDArrayIndex.all()}, f);
                labelsOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                        .put(new INDArrayIndex[]{NDArrayIndex.interval(0, tsLength), NDArrayIndex.all()}, listLabels.get(i));
                for( int j=tsLength; j<maxLength; j++ ){
                    featuresMask.put(i,j,0.0);
                    labelsMask.put(i,j,0.0);
                }
            }
        }

        cursor += listFeatures.size();
        if (inputColumns == -1) inputColumns = featuresOut.size(1);
        if (totalOutcomes == -1) totalOutcomes = labelsOut.size(1);
        DataSet ds = new DataSet(featuresOut, labelsOut, featuresMask, labelsMask);
        if(collectMetaData){
            ds.setExampleMetaData(meta);
        }
        if (preProcessor != null) preProcessor.preProcess(ds);
        return ds;
    }

    private DataSet nextMultipleSequenceReaders(int num) {
        List<INDArray> featureList = new ArrayList<>(num);
        List<INDArray> labelList = new ArrayList<>(num);
        List<RecordMetaData> meta = (collectMetaData ? new ArrayList<RecordMetaData>() : null);
        for (int i = 0; i < num && hasNext(); i++) {
            List<List<Writable>> featureSequence;
            List<List<Writable>> labelSequence;
            if (collectMetaData) {
                SequenceRecord f = recordReader.nextSequence();
                SequenceRecord l = labelsReader.nextSequence();
                featureSequence = f.getSequenceRecord();
                labelSequence = l.getSequenceRecord();
                meta.add(new RecordMetaDataComposable(f.getMetaData(), l.getMetaData()));
            } else {
                featureSequence = recordReader.sequenceRecord();
                labelSequence = labelsReader.sequenceRecord();
            }
            assertNonZeroLengthSequence(featureSequence, "features");
            assertNonZeroLengthSequence(labelSequence, "labels");

            INDArray features = getFeatures(featureSequence);
            INDArray labels = getLabels(labelSequence); //2d time series, with shape [timeSeriesLength,vectorSize]

            featureList.add(features);
            labelList.add(labels);
        }

        return nextMultipleSequenceReaders(featureList, labelList, meta);
    }

    private void assertNonZeroLengthSequence(List<?> sequence, String type) {
        if (sequence.size() == 0) {
            throw new ZeroLengthSequenceException(type);
        }
    }

    private DataSet nextMultipleSequenceReaders(List<INDArray> featureList, List<INDArray> labelList, List<RecordMetaData> meta ){

        //Convert 2d sequences/time series to 3d minibatch data
        INDArray featuresOut;
        INDArray labelsOut;
        INDArray featuresMask = null;
        INDArray labelsMask = null;
        if(alignmentMode == AlignmentMode.EQUAL_LENGTH) {
            int[] featureShape = new int[3];
            featureShape[0] = featureList.size();   //mini batch size
            featureShape[1] = featureList.get(0).size(1);   //example vector size
            featureShape[2] = featureList.get(0).size(0);   //time series/sequence length

            int[] labelShape = new int[3];
            labelShape[0] = labelList.size();
            labelShape[1] = labelList.get(0).size(1);   //label vector size
            labelShape[2] = labelList.get(0).size(0);   //time series/sequence length

            featuresOut = Nd4j.create(featureShape,'f');
            labelsOut = Nd4j.create(labelShape,'f');
            for (int i = 0; i < featureList.size(); i++) {
                featuresOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                        .assign(featureList.get(i));
                labelsOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                        .assign(labelList.get(i));
            }
        } else if( alignmentMode == AlignmentMode.ALIGN_START ){
            int longestTimeSeries = 0;
            for(INDArray features : featureList){
                longestTimeSeries = Math.max(features.size(0),longestTimeSeries);
            }
            for(INDArray labels : labelList ){
                longestTimeSeries = Math.max(labels.size(0),longestTimeSeries);
            }

            int[] featuresShape = new int[]{
                    featureList.size(), //# examples
                    featureList.get(0).size(1), //example vector size
                    longestTimeSeries};
            int[] labelsShape = new int[]{
                    labelList.size(), //# examples
                    labelList.get(0).size(1), //example vector size
                    longestTimeSeries};

            featuresOut = Nd4j.create(featuresShape,'f');
            labelsOut = Nd4j.create(labelsShape,'f');
            featuresMask = Nd4j.ones(featureList.size(),longestTimeSeries);
            labelsMask = Nd4j.ones(labelList.size(),longestTimeSeries);
            for (int i = 0; i < featureList.size(); i++) {
                INDArray f = featureList.get(i);
                INDArray l = labelList.get(i);

                //Again, permute is to put [timeSeriesLength,vectorSize] into a [vectorSize,timeSeriesLength] matrix
                featuresOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                        .put(new INDArrayIndex[]{NDArrayIndex.interval(0, f.size(0)), NDArrayIndex.all()}, f);
                labelsOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                        .put(new INDArrayIndex[]{NDArrayIndex.interval(0, l.size(0)), NDArrayIndex.all()}, l);
                for( int j=f.size(0); j<longestTimeSeries; j++ ){
                    featuresMask.putScalar(i,j,0.0);
                }
                for( int j=l.size(0); j<longestTimeSeries; j++ ){
                    labelsMask.putScalar(i,j,0.0);
                }
            }
        } else if( alignmentMode == AlignmentMode.ALIGN_END ){    //Align at end

            int longestTimeSeries = 0;
            for(INDArray features : featureList){
                longestTimeSeries = Math.max(features.size(0),longestTimeSeries);
            }
            for(INDArray labels : labelList ){
                longestTimeSeries = Math.max(labels.size(0),longestTimeSeries);
            }

            int[] featuresShape = new int[]{
                    featureList.size(), //# examples
                    featureList.get(0).size(1), //example vector size
                    longestTimeSeries};
            int[] labelsShape = new int[]{
                    labelList.size(), //# examples
                    labelList.get(0).size(1), //example vector size
                    longestTimeSeries};

            featuresOut = Nd4j.create(featuresShape,'f');
            labelsOut = Nd4j.create(labelsShape,'f');
            featuresMask = Nd4j.ones(featureList.size(), longestTimeSeries);
            labelsMask = Nd4j.ones(labelList.size(), longestTimeSeries);
            for (int i = 0; i < featureList.size(); i++) {
                INDArray f = featureList.get(i);
                INDArray l = labelList.get(i);

                int fLen = f.size(0);
                int lLen = l.size(0);

                if(fLen >= lLen){
                    //Align labels with end of features (features are longer)
                    featuresOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                            .put(new INDArrayIndex[]{NDArrayIndex.interval(0, fLen), NDArrayIndex.all()}, f);
                    labelsOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                            .put(new INDArrayIndex[]{NDArrayIndex.interval(fLen-lLen, fLen), NDArrayIndex.all()}, l);

                    for( int j=fLen; j<longestTimeSeries; j++ ){
                        featuresMask.putScalar(i,j,0.0);
                    }
                    //labels mask: component before labels
                    for( int j=0; j<fLen-lLen; j++ ){
                        labelsMask.putScalar(i,j,0.0);
                    }
                    //labels mask: component after labels
                    for( int j=fLen; j<longestTimeSeries; j++ ){
                        labelsMask.putScalar(i,j,0.0);
                    }
                } else {
                    //Align features with end of labels (labels are longer)
                    featuresOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                            .put(new INDArrayIndex[]{NDArrayIndex.interval(lLen-fLen, lLen), NDArrayIndex.all()}, f);
                    labelsOut.tensorAlongDimension(i, 1, 2).permutei(1,0)
                            .put(new INDArrayIndex[]{NDArrayIndex.interval(0, lLen), NDArrayIndex.all()}, l);

                    //features mask: component before features
                    for( int j=0; j<lLen-fLen; j++ ){
                        featuresMask.putScalar(i,j,0.0);
                    }
                    //features mask: component after features
                    for( int j=lLen; j<longestTimeSeries; j++ ){
                        featuresMask.putScalar(i,j,0.0);
                    }

                    //labels mask
                    for( int j=lLen; j<longestTimeSeries; j++ ){
                        labelsMask.putScalar(i,j,0.0);
                    }
                }
            }

        } else {
            throw new UnsupportedOperationException("Unknown alignment mode: " + alignmentMode);
        }

        cursor += featureList.size();
        if (inputColumns == -1) inputColumns = featuresOut.size(1);
        if (totalOutcomes == -1) totalOutcomes = labelsOut.size(1);
        DataSet ds = new DataSet(featuresOut, labelsOut, featuresMask, labelsMask);
        if(collectMetaData){
            ds.setExampleMetaData(meta);
        }
        if (preProcessor != null) preProcessor.preProcess(ds);
        return ds;
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int inputColumns() {
        if (inputColumns != -1) return inputColumns;
        preLoad();
        return inputColumns;
    }

    @Override
    public int totalOutcomes() {
        if (totalOutcomes != -1) return totalOutcomes;
        preLoad();
        return totalOutcomes;
    }

    private void preLoad() {
        stored = next();
        useStored = true;
        inputColumns = stored.getFeatureMatrix().size(1);
        totalOutcomes = stored.getLabels().size(1);
    }

    @Override
    public boolean resetSupported(){
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        recordReader.reset();
        if(labelsReader != null) labelsReader.reset();  //May be null for single seqRR case
        cursor = 0;
        stored = null;
        useStored = false;
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException("Not supported");
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

    private INDArray getFeatures(List<List<Writable>> features) {

        //Size of the record?
        int[] shape = new int[2]; //[timeSeriesLength,vectorSize]
        shape[0] = features.size();

        Iterator<List<Writable>> iter = features.iterator();

        int i = 0;
        INDArray out = null;
        while (iter.hasNext()) {
            List<Writable> step = iter.next();
            if (i == 0) {
                for( Writable w : step){
                    if(w instanceof NDArrayWritable){
                        shape[1] += ((NDArrayWritable) w).get().length();
                    } else {
                        shape[1]++;
                    }
                }
                out = Nd4j.create(shape,'f');
            }

            Iterator<Writable> timeStepIter = step.iterator();
            int f = 0;
            while (timeStepIter.hasNext()) {
                Writable current = timeStepIter.next();

                if(current instanceof NDArrayWritable){
                    //Array writable -> multiple values
                    INDArray arr = ((NDArrayWritable) current).get();
                    out.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(f,f+arr.length())}, arr);
                    f += arr.length();
                } else {
                    //Single value case
                    out.put(i, f++, current.toDouble());
                }
            }
            i++;
        }
        return out;
    }

    private INDArray getLabels(List<List<Writable>> labels) {
        //Size of the record?
        int[] shape = new int[2];   //[timeSeriesLength,vectorSize]
        shape[0] = labels.size();   //time series/sequence length

        Iterator<List<Writable>> iter = labels.iterator();

        int i = 0;
        INDArray out = null;
        while (iter.hasNext()) {
            List<Writable> step = iter.next();

            if (i == 0) {
                if (regression) {
                    for(Writable w : step){
                        if(w instanceof NDArrayWritable){
                            shape[1] += ((NDArrayWritable) w).get().length();
                        } else {
                            shape[1]++;
                        }
                    }
                } else {
                    shape[1] = numPossibleLabels;
                }
                out = Nd4j.create(shape,'f');
            }

            Iterator<Writable> timeStepIter = step.iterator();
            int f = 0;
            if (regression) {
                //Load all values
                while (timeStepIter.hasNext()) {
                    Writable current = timeStepIter.next();
                    if(current instanceof NDArrayWritable){
                        INDArray w = ((NDArrayWritable) current).get();
                        out.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(f,f+w.length())},w);
                        f += w.length();
                    } else {
                        out.put(i, f++, current.toDouble());
                    }
                }
            } else {
                //Expect a single value (index) -> convert to one-hot vector
                Writable value = timeStepIter.next();
                int idx = value.toInt();
                if(idx < 0 || idx >= numPossibleLabels){
                    throw new DL4JInvalidInputException("Invalid classification data: expect label value to be in range 0 to " +
                            (numPossibleLabels-1) + " inclusive (0 to numClasses-1, with numClasses=" + numPossibleLabels
                            + "); got label value of " + idx);
                }
                INDArray line = FeatureUtil.toOutcomeVector(idx, numPossibleLabels);
                out.getRow(i).assign(line);
            }

            i++;
        }
        return out;
    }

    private INDArray[] getFeaturesLabelsSingleReader(List<List<Writable>> input){
        Iterator<List<Writable>> iter = input.iterator();

        int i=0;
        INDArray features = null;
        INDArray labels = null;    //= Nd4j.zeros(input.size(), regression ? 1 : numPossibleLabels);

        int featureSize = 0;
        while(iter.hasNext()){
            List<Writable> step = iter.next();
            if (i == 0) {
                //First: determine the features size. Usually equal to the number of Writable objects, except when
                // one or more of the Writables is an INDArray (i.e., NDArrayWritable)
                int j=0;
                for(Writable w : step){
                    if(j++ != labelIndex) {
                        if (w instanceof NDArrayWritable) {
                            featureSize += ((NDArrayWritable) w).get().length();
                        } else {
                            featureSize += 1;
                        }
                    }
                }
                features = Nd4j.zeros( input.size(), featureSize);

                //Second: determine the output (labels) size.
                int labelSize;
                if(regression){
                    if(step.get(labelIndex) instanceof NDArrayWritable){
                        labelSize = ((NDArrayWritable) step.get(labelIndex)).get().length();
                    } else {
                        labelSize = 1;
                    }
                } else {
                    //Classification: integer -> one-hot
                    labelSize = numPossibleLabels;
                }
                labels = Nd4j.zeros(input.size(), labelSize);
            }

            Iterator<Writable> timeStepIter = step.iterator();
            int countIn = 0;
            int countFeatures = 0;
            while (timeStepIter.hasNext()) {
                Writable current = timeStepIter.next();
                if(countIn++ == labelIndex){
                    //label
                    if(regression){
                        if(current instanceof NDArrayWritable){
                            //Standard case
                            labels.putRow(i, ((NDArrayWritable) current).get());
                        } else {
                            labels.put(i,0,current.toDouble());
                        }
                    } else {
                        int idx = current.toInt();
                        if(idx < 0 || idx >= numPossibleLabels){
                            throw new DL4JInvalidInputException("Invalid classification data: expect label value (at label index column = " + labelIndex
                                    + ") to be in range 0 to " + (numPossibleLabels-1) + " inclusive (0 to numClasses-1, with numClasses=" + numPossibleLabels
                                    + "); got label value of " + current);
                        }
                        labels.putScalar(i,current.toInt(),1.0);    //Labels initialized as 0s
                    }
                } else {
                    //feature
                    if(current instanceof NDArrayWritable){
                        //NDArrayWritable: multiple values
                        INDArray w = ((NDArrayWritable) current).get();
                        int length = w.length();
                        features.put(new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.interval(countFeatures,countFeatures+length)}, w);
                        countFeatures += length;
                    } else {
                        //Standard case: single value
                        features.put(i, countFeatures++, current.toDouble());
                    }
                }
            }
            i++;
        }

        return new INDArray[]{features,labels};
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
        //Two cases: single vs. multiple reader...
        if(singleSequenceReaderMode){
            List<SequenceRecord> records = recordReader.loadSequenceFromMetaData(list);

            List<INDArray> listFeatures = new ArrayList<>(list.size());
            List<INDArray> listLabels = new ArrayList<>(list.size());
            int minLength = Integer.MAX_VALUE;
            int maxLength = Integer.MIN_VALUE;
            for(SequenceRecord sr : records){
                INDArray[] fl = getFeaturesLabelsSingleReader(sr.getSequenceRecord());
                listFeatures.add(fl[0]);
                listLabels.add(fl[1]);
                minLength = Math.min(minLength, fl[0].size(0));
                maxLength = Math.max(maxLength, fl[1].size(0));
            }

            return getSingleSequenceReader(listFeatures, listLabels, minLength, maxLength, list);
        } else {
            //Expect to get a RecordReaderMetaComposable here

            List<RecordMetaData> fMeta = new ArrayList<>();
            List<RecordMetaData> lMeta = new ArrayList<>();
            for(RecordMetaData m : list){
                RecordMetaDataComposable m2 = (RecordMetaDataComposable)m;
                fMeta.add(m2.getMeta()[0]);
                lMeta.add(m2.getMeta()[1]);
            }

            List<SequenceRecord> f = recordReader.loadSequenceFromMetaData(fMeta);
            List<SequenceRecord> l = labelsReader.loadSequenceFromMetaData(lMeta);

            List<INDArray> featureList = new ArrayList<>(fMeta.size());
            List<INDArray> labelList = new ArrayList<>(fMeta.size());

            for(int i=0; i<fMeta.size(); i++ ){
                featureList.add(getFeatures(f.get(i).getSequenceRecord()));
                labelList.add(getLabels(l.get(i).getSequenceRecord()));
            }

            return nextMultipleSequenceReaders(featureList, labelList, list);
        }
    }
}
