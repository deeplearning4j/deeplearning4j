/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets.datavec;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.ArrayUtils;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataComposableMap;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.*;

/**
 * RecordReaderMultiDataSetIterator: A {@link MultiDataSetIterator} for data from one or more RecordReaders and SequenceRecordReaders<br>
 * The idea: generate multiple inputs and multiple outputs from one or more Sequence/RecordReaders. Inputs and outputs
 * may be obtained from subsets of the RecordReader and SequenceRecordReaders columns (for examples, some inputs and outputs
 * as different columns in the same record/sequence); it is also possible to mix different types of data (for example, using both
 * RecordReaders and SequenceRecordReaders in the same RecordReaderMultiDataSetIterator).<br>
 * Uses a builder pattern ({@link RecordReaderMultiDataSetIterator.Builder} to specify the various
 * inputs and subsets.
 *
 * @author Alex Black
 */
public class RecordReaderMultiDataSetIterator implements MultiDataSetIterator {

    /**
     * When dealing with time series data of different lengths, how should we align the input/labels time series?
     * For equal length: use EQUAL_LENGTH
     * For sequence classification: use ALIGN_END
     */
    public enum AlignmentMode {
        EQUAL_LENGTH, ALIGN_START, ALIGN_END
    }

    private int batchSize;
    private AlignmentMode alignmentMode;
    private Map<String, RecordReader> recordReaders = new HashMap<>();
    private Map<String, SequenceRecordReader> sequenceRecordReaders = new HashMap<>();

    private List<SubsetDetails> inputs = new ArrayList<>();
    private List<SubsetDetails> outputs = new ArrayList<>();

    @Getter
    @Setter
    private boolean collectMetaData = false;

    private MultiDataSetPreProcessor preProcessor;

    private RecordReaderMultiDataSetIterator(Builder builder) {
        this.batchSize = builder.batchSize;
        this.alignmentMode = builder.alignmentMode;
        this.recordReaders = builder.recordReaders;
        this.sequenceRecordReaders = builder.sequenceRecordReaders;
        this.inputs.addAll(builder.inputs);
        this.outputs.addAll(builder.outputs);
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove not supported");
    }

    @Override
    public MultiDataSet next(int num) {
        if (!hasNext())
            throw new NoSuchElementException("No next elements");

        //First: load the next values from the RR / SeqRRs
        Map<String, List<List<Writable>>> nextRRVals = new HashMap<>();
        Map<String, List<List<List<Writable>>>> nextSeqRRVals = new HashMap<>();
        List<RecordMetaDataComposableMap> nextMetas =
                        (collectMetaData ? new ArrayList<RecordMetaDataComposableMap>() : null);


        for (Map.Entry<String, RecordReader> entry : recordReaders.entrySet()) {
            RecordReader rr = entry.getValue();
            List<List<Writable>> writables = new ArrayList<>(num);
            for (int i = 0; i < num && rr.hasNext(); i++) {
                List<Writable> record;
                if (collectMetaData) {
                    Record r = rr.nextRecord();
                    record = r.getRecord();
                    if (nextMetas.size() <= i) {
                        nextMetas.add(new RecordMetaDataComposableMap(new HashMap<String, RecordMetaData>()));
                    }
                    RecordMetaDataComposableMap map = nextMetas.get(i);
                    map.getMeta().put(entry.getKey(), r.getMetaData());
                } else {
                    record = rr.next();
                }
                writables.add(record);
            }

            nextRRVals.put(entry.getKey(), writables);
        }

        for (Map.Entry<String, SequenceRecordReader> entry : sequenceRecordReaders.entrySet()) {
            SequenceRecordReader rr = entry.getValue();
            List<List<List<Writable>>> writables = new ArrayList<>(num);
            for (int i = 0; i < num && rr.hasNext(); i++) {
                List<List<Writable>> sequence;
                if (collectMetaData) {
                    SequenceRecord r = rr.nextSequence();
                    sequence = r.getSequenceRecord();
                    if (nextMetas.size() <= i) {
                        nextMetas.add(new RecordMetaDataComposableMap(new HashMap<String, RecordMetaData>()));
                    }
                    RecordMetaDataComposableMap map = nextMetas.get(i);
                    map.getMeta().put(entry.getKey(), r.getMetaData());
                } else {
                    sequence = rr.sequenceRecord();
                }
                writables.add(sequence);
            }

            nextSeqRRVals.put(entry.getKey(), writables);
        }

        return nextMultiDataSet(nextRRVals, nextSeqRRVals, nextMetas);
    }

    private MultiDataSet nextMultiDataSet(Map<String, List<List<Writable>>> nextRRVals,
                    Map<String, List<List<List<Writable>>>> nextSeqRRVals,
                    List<RecordMetaDataComposableMap> nextMetas) {
        int minExamples = Integer.MAX_VALUE;
        for (List<List<Writable>> exampleData : nextRRVals.values()) {
            minExamples = Math.min(minExamples, exampleData.size());
        }
        for (List<List<List<Writable>>> exampleData : nextSeqRRVals.values()) {
            minExamples = Math.min(minExamples, exampleData.size());
        }


        if (minExamples == Integer.MAX_VALUE)
            throw new RuntimeException("Error occurred during data set generation: no readers?"); //Should never happen

        //In order to align data at the end (for each example individually), we need to know the length of the
        // longest time series for each example
        int[] longestSequence = null;
        if (alignmentMode == AlignmentMode.ALIGN_END) {
            longestSequence = new int[minExamples];
            for (Map.Entry<String, List<List<List<Writable>>>> entry : nextSeqRRVals.entrySet()) {
                List<List<List<Writable>>> list = entry.getValue();
                for (int i = 0; i < list.size() && i < minExamples; i++) {
                    longestSequence[i] = Math.max(longestSequence[i], list.get(i).size());
                }
            }
        }

        //Second: create the input arrays
        //To do this, we need to know longest time series length, so we can do padding
        int longestTS = -1;
        if (alignmentMode != AlignmentMode.EQUAL_LENGTH) {
            for (Map.Entry<String, List<List<List<Writable>>>> entry : nextSeqRRVals.entrySet()) {
                List<List<List<Writable>>> list = entry.getValue();
                for (List<List<Writable>> c : list) {
                    longestTS = Math.max(longestTS, c.size());
                }
            }
        }

        INDArray[] inputArrs = new INDArray[inputs.size()];
        INDArray[] inputArrMasks = new INDArray[inputs.size()];
        boolean inputMasks = false;
        int i = 0;
        for (SubsetDetails d : inputs) {
            if (nextRRVals.containsKey(d.readerName)) {
                //Standard reader
                List<List<Writable>> list = nextRRVals.get(d.readerName);
                inputArrs[i] = convertWritables(list, minExamples, d);
            } else {
                //Sequence reader
                List<List<List<Writable>>> list = nextSeqRRVals.get(d.readerName);
                Pair<INDArray, INDArray> p = convertWritablesSequence(list, minExamples, longestTS, d, longestSequence);
                inputArrs[i] = p.getFirst();
                inputArrMasks[i] = p.getSecond();
                if (inputArrMasks[i] != null)
                    inputMasks = true;
            }
            i++;
        }
        if (!inputMasks)
            inputArrMasks = null;


        //Third: create the outputs
        INDArray[] outputArrs = new INDArray[outputs.size()];
        INDArray[] outputArrMasks = new INDArray[outputs.size()];
        boolean outputMasks = false;
        i = 0;
        for (SubsetDetails d : outputs) {
            if (nextRRVals.containsKey(d.readerName)) {
                //Standard reader
                List<List<Writable>> list = nextRRVals.get(d.readerName);
                outputArrs[i] = convertWritables(list, minExamples, d);
            } else {
                //Sequence reader
                List<List<List<Writable>>> list = nextSeqRRVals.get(d.readerName);
                Pair<INDArray, INDArray> p = convertWritablesSequence(list, minExamples, longestTS, d, longestSequence);
                outputArrs[i] = p.getFirst();
                outputArrMasks[i] = p.getSecond();
                if (outputArrMasks[i] != null)
                    outputMasks = true;
            }
            i++;
        }
        if (!outputMasks)
            outputArrMasks = null;

        MultiDataSet mds =
                        new org.nd4j.linalg.dataset.MultiDataSet(inputArrs, outputArrs, inputArrMasks, outputArrMasks);
        if (collectMetaData) {
            mds.setExampleMetaData(nextMetas);
        }
        if (preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    private INDArray convertWritables(List<List<Writable>> list, int minValues, SubsetDetails details) {
        INDArray arr;
        if (details.entireReader) {
            if (list.get(0).size() == 1 && list.get(0).get(0) instanceof NDArrayWritable) {
                //Special case: single NDArrayWritable...
                INDArray temp = ((NDArrayWritable) list.get(0).get(0)).get();
                int[] shape = ArrayUtils.clone(temp.shape());
                shape[0] = minValues;
                arr = Nd4j.create(shape);
            } else {
                arr = Nd4j.create(minValues, list.get(0).size());
            }
        } else if (details.oneHot) {
            arr = Nd4j.zeros(minValues, details.oneHotNumClasses);
        } else {
            if (details.subsetStart == details.subsetEndInclusive
                            && list.get(0).get(details.subsetStart) instanceof NDArrayWritable) {
                //Special case: single NDArrayWritable (example: ImageRecordReader)
                INDArray temp = ((NDArrayWritable) list.get(0).get(details.subsetStart)).get();
                int[] shape = ArrayUtils.clone(temp.shape());
                shape[0] = minValues;
                arr = Nd4j.create(shape);
            } else {
                arr = Nd4j.create(minValues, details.subsetEndInclusive - details.subsetStart + 1);
            }
        }

        for (int i = 0; i < minValues; i++) {
            List<Writable> c = list.get(i);
            if (details.entireReader) {
                //Convert entire reader contents, without modification
                int j = 0;
                for (Writable w : c) {
                    try {
                        arr.putScalar(i, j, w.toDouble());
                    } catch (UnsupportedOperationException e) {
                        // This isn't a scalar, so check if we got an array already
                        if (w instanceof NDArrayWritable) {
                            putExample(arr, ((NDArrayWritable) w).get(), i);
                        } else {
                            throw e;
                        }
                    }
                    j++;
                }
            } else if (details.oneHot) {
                //Convert a single column to a one-hot representation
                Writable w = c.get(details.subsetStart);
                //Index of class
                arr.putScalar(i, w.toInt(), 1.0);
            } else {
                //Convert a subset of the columns

                //Special case: subsetStart == subsetEndInclusive && NDArrayWritable. Example: ImageRecordReader
                if (details.subsetStart == details.subsetEndInclusive
                                && (c.get(details.subsetStart) instanceof NDArrayWritable)) {
                    putExample(arr, ((NDArrayWritable) c.get(details.subsetStart)).get(), i);
                } else {

                    Iterator<Writable> iter = c.iterator();
                    for (int j = 0; j < details.subsetStart; j++)
                        iter.next();
                    int k = 0;
                    for (int j = details.subsetStart; j <= details.subsetEndInclusive; j++) {
                        Writable w = iter.next();
                        try {
                            arr.putScalar(i, k, w.toDouble());
                        } catch (UnsupportedOperationException e) {
                            // This isn't a scalar, so check if we got an array already
                            if (w instanceof NDArrayWritable) {
                                putExample(arr, ((NDArrayWritable) w).get(), i);
                            } else {
                                throw e;
                            }
                        }
                        k++;
                    }
                }
            }
        }

        return arr;
    }

    private void putExample(INDArray arr, INDArray singleExample, int exampleIdx) {
        switch (arr.rank()) {
            case 2:
                arr.put(new INDArrayIndex[] {NDArrayIndex.point(exampleIdx), NDArrayIndex.all()}, singleExample);
                break;
            case 3:
                arr.put(new INDArrayIndex[] {NDArrayIndex.point(exampleIdx), NDArrayIndex.all(), NDArrayIndex.all()},
                                singleExample);
                break;
            case 4:
                arr.put(new INDArrayIndex[] {NDArrayIndex.point(exampleIdx), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all()}, singleExample);
                break;
            default:
                throw new RuntimeException("Unexpected rank: " + arr.rank());
        }
    }

    /**
     * Convert the writables to a sequence (3d) data set, and also return the mask array (if necessary)
     */
    private Pair<INDArray, INDArray> convertWritablesSequence(List<List<List<Writable>>> list, int minValues,
                    int maxTSLength, SubsetDetails details, int[] longestSequence) {
        if (maxTSLength == -1)
            maxTSLength = list.get(0).size();
        INDArray arr;
        if (details.entireReader) {
            int size = list.get(0).iterator().next().size();
            arr = Nd4j.create(new int[] {minValues, size, maxTSLength}, 'f');
        } else if (details.oneHot)
            arr = Nd4j.create(new int[] {minValues, details.oneHotNumClasses, maxTSLength}, 'f');
        else
            arr = Nd4j.create(new int[] {minValues, details.subsetEndInclusive - details.subsetStart + 1, maxTSLength},
                            'f');

        boolean needMaskArray = false;
        for (List<List<Writable>> c : list) {
            if (c.size() < maxTSLength)
                needMaskArray = true;
        }

        if(needMaskArray && alignmentMode == AlignmentMode.EQUAL_LENGTH ){
            throw new UnsupportedOperationException("Alignment mode is set to EQUAL_LENGTH but variable length data was "
                    + "encountered. Use AlignmentMode.ALIGN_START or AlignmentMode.ALIGN_END with variable length data");
        }

        INDArray maskArray;
        if (needMaskArray)
            maskArray = Nd4j.ones(minValues, maxTSLength);
        else
            maskArray = null;

        for (int i = 0; i < minValues; i++) {
            List<List<Writable>> sequence = list.get(i);

            //Offset for alignment:
            int startOffset;
            if (alignmentMode == AlignmentMode.ALIGN_START || alignmentMode == AlignmentMode.EQUAL_LENGTH) {
                startOffset = 0;
            } else {
                //Align end
                //Only practical differences here are: (a) offset, and (b) masking
                startOffset = longestSequence[i] - sequence.size();
            }

            int t = 0;
            int k;
            for (List<Writable> timeStep : sequence) {
                k = startOffset + t++;

                if (details.entireReader) {
                    //Convert entire reader contents, without modification
                    Iterator<Writable> iter = timeStep.iterator();
                    int j = 0;
                    while (iter.hasNext()) {
                        Writable w = iter.next();
                        try {
                            arr.putScalar(i, j, k, w.toDouble());
                        } catch (UnsupportedOperationException e) {
                            // This isn't a scalar, so check if we got an array already
                            if (w instanceof NDArrayWritable) {
                                arr.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(k)).putRow(0,
                                                ((NDArrayWritable) w).get());
                            } else {
                                throw e;
                            }
                        }
                        j++;
                    }
                } else if (details.oneHot) {
                    //Convert a single column to a one-hot representation
                    Writable w = null;
                    if (timeStep instanceof List)
                        w = ((List<Writable>) timeStep).get(details.subsetStart);
                    else {
                        Iterator<Writable> iter = timeStep.iterator();
                        for (int x = 0; x <= details.subsetStart; x++)
                            w = iter.next();
                    }
                    int classIdx = w.toInt();
                    arr.putScalar(i, classIdx, k, 1.0);
                } else {
                    //Convert a subset of the columns...
                    Iterator<Writable> iter = timeStep.iterator();
                    for (int j = 0; j < details.subsetStart; j++)
                        iter.next();
                    int l = 0;
                    for (int j = details.subsetStart; j <= details.subsetEndInclusive; j++) {
                        Writable w = iter.next();
                        try {
                            arr.putScalar(i, l++, k, w.toDouble());
                        } catch (UnsupportedOperationException e) {
                            // This isn't a scalar, so check if we got an array already
                            if (w instanceof NDArrayWritable) {
                                arr.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(k)).putRow(0,
                                                ((NDArrayWritable) w).get()
                                                                .get(NDArrayIndex.all(), NDArrayIndex.interval(
                                                                                details.subsetStart, details.subsetEndInclusive
                                                                                                + 1)));
                            } else {
                                throw e;
                            }
                        }
                    }
                }
            }

            //For any remaining time steps: set mask array to 0 (just padding)
            if (needMaskArray) {
                //Masking array entries at start (for align end)
                if (alignmentMode == AlignmentMode.ALIGN_END) {
                    for (int t2 = 0; t2 < startOffset; t2++) {
                        maskArray.putScalar(i, t2, 0.0);
                    }
                }

                //Masking array entries at end (for align start)
                if (alignmentMode == AlignmentMode.ALIGN_START) {
                    for (int t2 = t; t2 < maxTSLength; t2++) {
                        maskArray.putScalar(i, t2, 0.0);
                    }
                }
            }
        }


        return new Pair<>(arr, maskArray);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
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
        for (RecordReader rr : recordReaders.values())
            rr.reset();
        for (SequenceRecordReader rr : sequenceRecordReaders.values())
            rr.reset();
    }

    @Override
    public boolean hasNext() {
        for (RecordReader rr : recordReaders.values())
            if (!rr.hasNext())
                return false;
        for (SequenceRecordReader rr : sequenceRecordReaders.values())
            if (!rr.hasNext())
                return false;
        return true;
    }


    public static class Builder {

        private int batchSize;
        private AlignmentMode alignmentMode = AlignmentMode.ALIGN_START;
        private Map<String, RecordReader> recordReaders = new HashMap<>();
        private Map<String, SequenceRecordReader> sequenceRecordReaders = new HashMap<>();

        private List<SubsetDetails> inputs = new ArrayList<>();
        private List<SubsetDetails> outputs = new ArrayList<>();

        /**
         * @param batchSize The batch size for the RecordReaderMultiDataSetIterator
         */
        public Builder(int batchSize) {
            this.batchSize = batchSize;
        }

        /**
         * Add a RecordReader for use in .addInput(...) or .addOutput(...)
         *
         * @param readerName   Name of the reader (for later reference)
         * @param recordReader RecordReader
         */
        public Builder addReader(String readerName, RecordReader recordReader) {
            recordReaders.put(readerName, recordReader);
            return this;
        }

        /**
         * Add a SequenceRecordReader for use in .addInput(...) or .addOutput(...)
         *
         * @param seqReaderName   Name of the sequence reader (for later reference)
         * @param seqRecordReader SequenceRecordReader
         */
        public Builder addSequenceReader(String seqReaderName, SequenceRecordReader seqRecordReader) {
            sequenceRecordReaders.put(seqReaderName, seqRecordReader);
            return this;
        }

        /**
         * Set the sequence alignment mode for all sequences
         */
        public Builder sequenceAlignmentMode(AlignmentMode alignmentMode) {
            this.alignmentMode = alignmentMode;
            return this;
        }

        /**
         * Set as an input, the entire contents (all columns) of the RecordReader or SequenceRecordReader
         */
        public Builder addInput(String readerName) {
            inputs.add(new SubsetDetails(readerName, true, false, -1, -1, -1));
            return this;
        }

        /**
         * Set as an input, a subset of the specified RecordReader or SequenceRecordReader
         *
         * @param readerName  Name of the reader
         * @param columnFirst First column index, inclusive
         * @param columnLast  Last column index, inclusive
         */
        public Builder addInput(String readerName, int columnFirst, int columnLast) {
            inputs.add(new SubsetDetails(readerName, false, false, -1, columnFirst, columnLast));
            return this;
        }

        /**
         * Add as an input a single column from the specified RecordReader / SequenceRecordReader
         * The assumption is that the specified column contains integer values in range 0..numClasses-1;
         * this integer will be converted to a one-hot representation
         *
         * @param readerName Name of the RecordReader or SequenceRecordReader
         * @param column     Column that contains the index
         * @param numClasses Total number of classes
         */
        public Builder addInputOneHot(String readerName, int column, int numClasses) {
            inputs.add(new SubsetDetails(readerName, false, true, numClasses, column, -1));
            return this;
        }

        /**
         * Set as an output, the entire contents (all columns) of the RecordReader or SequenceRecordReader
         */
        public Builder addOutput(String readerName) {
            outputs.add(new SubsetDetails(readerName, true, false, -1, -1, -1));
            return this;
        }

        /**
         * Add an output, with a subset of the columns from the named RecordReader or SequenceRecordReader
         *
         * @param readerName  Name of the reader
         * @param columnFirst First column index
         * @param columnLast  Last column index (inclusive)
         */
        public Builder addOutput(String readerName, int columnFirst, int columnLast) {
            outputs.add(new SubsetDetails(readerName, false, false, -1, columnFirst, columnLast));
            return this;
        }

        /**
         * An an output, where the output is taken from a single column from the specified RecordReader / SequenceRecordReader
         * The assumption is that the specified column contains integer values in range 0..numClasses-1;
         * this integer will be converted to a one-hot representation (usually for classification)
         *
         * @param readerName Name of the RecordReader / SequenceRecordReader
         * @param column     index of the column
         * @param numClasses Number of classes
         */
        public Builder addOutputOneHot(String readerName, int column, int numClasses) {
            outputs.add(new SubsetDetails(readerName, false, true, numClasses, column, -1));
            return this;
        }

        /**
         * Create the RecordReaderMultiDataSetIterator
         */
        public RecordReaderMultiDataSetIterator build() {
            //Validate input:
            if (recordReaders.isEmpty() && sequenceRecordReaders.isEmpty()) {
                throw new IllegalStateException("Cannot construct RecordReaderMultiDataSetIterator with no readers");
            }

            if (batchSize <= 0)
                throw new IllegalStateException(
                                "Cannot construct RecordReaderMultiDataSetIterator with batch size <= 0");

            if (inputs.isEmpty() && outputs.isEmpty()) {
                throw new IllegalStateException(
                                "Cannot construct RecordReaderMultiDataSetIterator with no inputs/outputs");
            }

            for (SubsetDetails ssd : inputs) {
                if (!recordReaders.containsKey(ssd.readerName) && !sequenceRecordReaders.containsKey(ssd.readerName)) {
                    throw new IllegalStateException(
                                    "Invalid input name: \"" + ssd.readerName + "\" - no reader found with this name");
                }
            }

            for (SubsetDetails ssd : outputs) {
                if (!recordReaders.containsKey(ssd.readerName) && !sequenceRecordReaders.containsKey(ssd.readerName)) {
                    throw new IllegalStateException(
                                    "Invalid output name: \"" + ssd.readerName + "\" - no reader found with this name");
                }
            }

            return new RecordReaderMultiDataSetIterator(this);
        }
    }

    /**
     * Load a single example to a DataSet, using the provided RecordMetaData.
     * Note that it is more efficient to load multiple instances at once, using {@link #loadFromMetaData(List)}
     *
     * @param recordMetaData RecordMetaData to load from. Should have been produced by the given record reader
     * @return DataSet with the specified example
     * @throws IOException If an error occurs during loading of the data
     */
    public MultiDataSet loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
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
    public MultiDataSet loadFromMetaData(List<RecordMetaData> list) throws IOException {
        //First: load the next values from the RR / SeqRRs
        Map<String, List<List<Writable>>> nextRRVals = new HashMap<>();
        Map<String, List<List<List<Writable>>>> nextSeqRRVals = new HashMap<>();
        List<RecordMetaDataComposableMap> nextMetas =
                        (collectMetaData ? new ArrayList<RecordMetaDataComposableMap>() : null);


        for (Map.Entry<String, RecordReader> entry : recordReaders.entrySet()) {
            RecordReader rr = entry.getValue();

            List<RecordMetaData> thisRRMeta = new ArrayList<>();
            for (RecordMetaData m : list) {
                RecordMetaDataComposableMap m2 = (RecordMetaDataComposableMap) m;
                thisRRMeta.add(m2.getMeta().get(entry.getKey()));
            }

            List<Record> fromMeta = rr.loadFromMetaData(thisRRMeta);
            List<List<Writable>> writables = new ArrayList<>(list.size());
            for (Record r : fromMeta) {
                writables.add(r.getRecord());
            }

            nextRRVals.put(entry.getKey(), writables);
        }

        for (Map.Entry<String, SequenceRecordReader> entry : sequenceRecordReaders.entrySet()) {
            SequenceRecordReader rr = entry.getValue();

            List<RecordMetaData> thisRRMeta = new ArrayList<>();
            for (RecordMetaData m : list) {
                RecordMetaDataComposableMap m2 = (RecordMetaDataComposableMap) m;
                thisRRMeta.add(m2.getMeta().get(entry.getKey()));
            }

            List<SequenceRecord> fromMeta = rr.loadSequenceFromMetaData(thisRRMeta);
            List<List<List<Writable>>> writables = new ArrayList<>(list.size());
            for (SequenceRecord r : fromMeta) {
                writables.add(r.getSequenceRecord());
            }

            nextSeqRRVals.put(entry.getKey(), writables);
        }

        return nextMultiDataSet(nextRRVals, nextSeqRRVals, nextMetas);

    }

    @AllArgsConstructor
    private static class SubsetDetails {
        private final String readerName;
        private final boolean entireReader;
        private final boolean oneHot;
        private final int oneHotNumClasses;
        private final int subsetStart;
        private final int subsetEndInclusive;
    }
}
