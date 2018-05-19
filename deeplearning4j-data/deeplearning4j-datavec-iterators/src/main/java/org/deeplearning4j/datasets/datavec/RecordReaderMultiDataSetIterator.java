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
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataComposableMap;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.deeplearning4j.datasets.datavec.exception.ZeroLengthSequenceException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.io.Serializable;
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
@Getter
public class RecordReaderMultiDataSetIterator implements MultiDataSetIterator, Serializable {

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

    private boolean timeSeriesRandomOffset = false;
    private Random timeSeriesRandomOffsetRng;

    private MultiDataSetPreProcessor preProcessor;

    private boolean resetSupported = true;

    private RecordReaderMultiDataSetIterator(Builder builder) {
        this.batchSize = builder.batchSize;
        this.alignmentMode = builder.alignmentMode;
        this.recordReaders = builder.recordReaders;
        this.sequenceRecordReaders = builder.sequenceRecordReaders;
        this.inputs.addAll(builder.inputs);
        this.outputs.addAll(builder.outputs);
        this.timeSeriesRandomOffset = builder.timeSeriesRandomOffset;
        if (this.timeSeriesRandomOffset) {
            timeSeriesRandomOffsetRng = new Random(builder.timeSeriesRandomOffsetSeed);
        }


        if(recordReaders != null){
            for(RecordReader rr : recordReaders.values()){
                resetSupported &= rr.resetSupported();
            }
        }
        if(sequenceRecordReaders != null){
            for(SequenceRecordReader srr : sequenceRecordReaders.values()){
                resetSupported &= srr.resetSupported();
            }
        }
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
        Map<String, List<INDArray>> nextRRValsBatched = null;
        Map<String, List<List<List<Writable>>>> nextSeqRRVals = new HashMap<>();
        List<RecordMetaDataComposableMap> nextMetas =
                        (collectMetaData ? new ArrayList<RecordMetaDataComposableMap>() : null);


        for (Map.Entry<String, RecordReader> entry : recordReaders.entrySet()) {
            RecordReader rr = entry.getValue();
            if (!collectMetaData && rr.batchesSupported()) {
                //Batch case, for efficiency: ImageRecordReader etc
                List<List<Writable>> batchWritables = rr.next(num);

                List<INDArray> batch;
                if(batchWritables instanceof NDArrayRecordBatch){
                    //ImageRecordReader etc case
                    batch = ((NDArrayRecordBatch)batchWritables).getArrays();
                } else {
                    batchWritables = filterRequiredColumns(entry.getKey(), batchWritables);
                    batch = new ArrayList<>();
                    List<Writable> temp = new ArrayList<>();
                    int sz = batchWritables.get(0).size();
                    for( int i=0; i<sz; i++ ){
                        temp.clear();
                        for( int j=0; j<batchWritables.size(); j++ ){
                            temp.add(batchWritables.get(j).get(i));
                        }
                        batch.add(RecordConverter.toMinibatchArray(temp));
                    }
                }

                if (nextRRValsBatched == null) {
                    nextRRValsBatched = new HashMap<>();
                }
                nextRRValsBatched.put(entry.getKey(), batch);
            } else {
                //Standard case
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

        return nextMultiDataSet(nextRRVals, nextRRValsBatched, nextSeqRRVals, nextMetas);
    }

    //Filter out the required columns before conversion. This is to avoid trying to convert String etc columns
    private List<List<Writable>> filterRequiredColumns(String readerName, List<List<Writable>> list){

        //Options: (a) entire reader
        //(b) one or more subsets

        boolean entireReader = false;
        List<SubsetDetails> subsetList = null;
        int max = -1;
        int min = Integer.MAX_VALUE;
        for(List<SubsetDetails> sdList : Arrays.asList(inputs, outputs)) {
            for (SubsetDetails sd : sdList) {
                if (readerName.equals(sd.readerName)) {
                    if (sd.entireReader) {
                        entireReader = true;
                        break;
                    } else {
                        if (subsetList == null) {
                            subsetList = new ArrayList<>();
                        }
                        subsetList.add(sd);
                        max = Math.max(max, sd.subsetEndInclusive);
                        min = Math.min(min, sd.subsetStart);
                    }
                }
            }
        }

        if(entireReader){
            //No filtering required
            return list;
        } else if(subsetList == null){
            throw new IllegalStateException("Found no usages of reader: " + readerName);
        } else {
            //we need some - but not all - columns
            boolean[] req = new boolean[max+1];
            for(SubsetDetails sd : subsetList){
                for( int i=sd.subsetStart; i<= sd.subsetEndInclusive; i++ ){
                    req[i] = true;
                }
            }

            List<List<Writable>> out = new ArrayList<>();
            IntWritable zero = new IntWritable(0);
            for(List<Writable> l : list){
                List<Writable> lNew = new ArrayList<>(l.size());
                for(int i=0; i<l.size(); i++ ){
                    if(i >= req.length || !req[i]){
                        lNew.add(zero);
                    } else {
                        lNew.add(l.get(i));
                    }
                }
                out.add(lNew);
            }
            return out;
        }
    }

    public MultiDataSet nextMultiDataSet(Map<String, List<List<Writable>>> nextRRVals,
                    Map<String, List<INDArray>> nextRRValsBatched,
                    Map<String, List<List<List<Writable>>>> nextSeqRRVals,
                    List<RecordMetaDataComposableMap> nextMetas) {
        int minExamples = Integer.MAX_VALUE;
        for (List<List<Writable>> exampleData : nextRRVals.values()) {
            minExamples = Math.min(minExamples, exampleData.size());
        }
        if (nextRRValsBatched != null) {
            for (List<INDArray> exampleData : nextRRValsBatched.values()) {
                //Assume all NDArrayWritables here
                for (INDArray w : exampleData) {
                    val n = w.size(0);

                    // FIXME: int cast
                    minExamples = (int) Math.min(minExamples, n);
                }
            }
        }
        for (List<List<List<Writable>>> exampleData : nextSeqRRVals.values()) {
            minExamples = Math.min(minExamples, exampleData.size());
        }


        if (minExamples == Integer.MAX_VALUE)
            throw new RuntimeException("Error occurred during data set generation: no readers?"); //Should never happen

        //In order to align data at the end (for each example individually), we need to know the length of the
        // longest time series for each example
        int[] longestSequence = null;
        if (timeSeriesRandomOffset || alignmentMode == AlignmentMode.ALIGN_END) {
            longestSequence = new int[minExamples];
            for (Map.Entry<String, List<List<List<Writable>>>> entry : nextSeqRRVals.entrySet()) {
                List<List<List<Writable>>> list = entry.getValue();
                for (int i = 0; i < list.size() && i < minExamples; i++) {
                    longestSequence[i] = Math.max(longestSequence[i], list.get(i).size());
                }
            }
        }

        //Second: create the input/feature arrays
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
        long rngSeed = (timeSeriesRandomOffset ? timeSeriesRandomOffsetRng.nextLong() : -1);
        Pair<INDArray[], INDArray[]> features = convertFeaturesOrLabels(new INDArray[inputs.size()],
                        new INDArray[inputs.size()], inputs, minExamples, nextRRVals, nextRRValsBatched, nextSeqRRVals,
                        longestTS, longestSequence, rngSeed);


        //Third: create the outputs/labels
        Pair<INDArray[], INDArray[]> labels = convertFeaturesOrLabels(new INDArray[outputs.size()],
                        new INDArray[outputs.size()], outputs, minExamples, nextRRVals, nextRRValsBatched,
                        nextSeqRRVals, longestTS, longestSequence, rngSeed);



        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(features.getFirst(), labels.getFirst(),
                        features.getSecond(), labels.getSecond());
        if (collectMetaData) {
            mds.setExampleMetaData(nextMetas);
        }
        if (preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    private Pair<INDArray[], INDArray[]> convertFeaturesOrLabels(INDArray[] featuresOrLabels, INDArray[] masks,
                    List<SubsetDetails> subsetDetails, int minExamples, Map<String, List<List<Writable>>> nextRRVals,
                    Map<String, List<INDArray>> nextRRValsBatched,
                    Map<String, List<List<List<Writable>>>> nextSeqRRVals, int longestTS, int[] longestSequence,
                    long rngSeed) {
        boolean hasMasks = false;
        int i = 0;

        for (SubsetDetails d : subsetDetails) {
            if (nextRRValsBatched != null && nextRRValsBatched.containsKey(d.readerName)) {
                //Standard reader, but batch ops
                featuresOrLabels[i] = convertWritablesBatched(nextRRValsBatched.get(d.readerName), d);
            } else if (nextRRVals.containsKey(d.readerName)) {
                //Standard reader
                List<List<Writable>> list = nextRRVals.get(d.readerName);
                featuresOrLabels[i] = convertWritables(list, minExamples, d);
            } else {
                //Sequence reader
                List<List<List<Writable>>> list = nextSeqRRVals.get(d.readerName);
                Pair<INDArray, INDArray> p =
                                convertWritablesSequence(list, minExamples, longestTS, d, longestSequence, rngSeed);
                featuresOrLabels[i] = p.getFirst();
                masks[i] = p.getSecond();
                if (masks[i] != null)
                    hasMasks = true;
            }
            i++;
        }

        return new Pair<>(featuresOrLabels, hasMasks ? masks : null);
    }

    private INDArray convertWritablesBatched(List<INDArray> list, SubsetDetails details) {
        INDArray arr;
        if (details.entireReader) {
            if (list.size() == 1) {
                arr = list.get(0);
            } else {
                //Need to concat column vectors
                INDArray[] asArray = list.toArray(new INDArray[list.size()]);
                arr = Nd4j.concat(1, asArray);
            }
        } else if (details.subsetStart == details.subsetEndInclusive || details.oneHot) {
            arr = list.get(details.subsetStart);
        } else {
            //Concat along dimension 1
            int count = details.subsetEndInclusive - details.subsetStart + 1;
            INDArray[] temp = new INDArray[count];
            int x = 0;
            for( int i=details.subsetStart; i<= details.subsetEndInclusive; i++){
                temp[x++] = list.get(i);
            }
            arr = Nd4j.concat(1, temp);
        }

        if (!details.oneHot || arr.size(1) == details.oneHotNumClasses) {
            //Not one-hot: no conversion required
            //Also, ImageRecordReader already does the one-hot conversion internally
            return arr;
        }

        //Do one-hot conversion
        if (arr.size(1) != 1) {
            throw new UnsupportedOperationException("Cannot do conversion to one hot using batched reader: "
                            + details.oneHotNumClasses + " output classes, but array.size(1) is " + arr.size(1)
                            + " (must be equal to 1 or numClasses = " + details.oneHotNumClasses + ")");
        }

        val n = arr.size(0);
        INDArray out = Nd4j.create(n, details.oneHotNumClasses);
        for (int i = 0; i < n; i++) {
            int v = arr.getInt(i, 0);
            out.putScalar(i, v, 1.0);
        }

        return out;
    }

    private int countLength(List<Writable> list) {
        return countLength(list, 0, list.size() - 1);
    }

    private int countLength(List<Writable> list, int from, int to) {
        int length = 0;
        for (int i = from; i <= to; i++) {
            Writable w = list.get(i);
            if (w instanceof NDArrayWritable) {
                INDArray a = ((NDArrayWritable) w).get();
                if (!a.isRowVectorOrScalar()) {
                    throw new UnsupportedOperationException("Multiple writables present but NDArrayWritable is "
                                    + "not a row vector. Can only concat row vectors with other writables. Shape: "
                                    + Arrays.toString(a.shape()));
                }
                length += a.length();
            } else {
                //Assume all others are single value
                length++;
            }
        }

        return length;
    }

    private INDArray convertWritables(List<List<Writable>> list, int minValues, SubsetDetails details) {
        try{
            return convertWritablesHelper(list, minValues, details);
        } catch (NumberFormatException e) {
            throw new RuntimeException("Error parsing data (writables) from record readers - value is non-numeric", e);
        } catch(IllegalStateException e){
            throw e;
        } catch (Throwable t){
            throw new RuntimeException("Error parsing data (writables) from record readers", t);
        }
    }

    private INDArray convertWritablesHelper(List<List<Writable>> list, int minValues, SubsetDetails details) {
        INDArray arr;
        if (details.entireReader) {
            if (list.get(0).size() == 1 && list.get(0).get(0) instanceof NDArrayWritable) {
                //Special case: single NDArrayWritable...
                INDArray temp = ((NDArrayWritable) list.get(0).get(0)).get();
                val shape = ArrayUtils.clone(temp.shape());
                shape[0] = minValues;
                arr = Nd4j.create(shape);
            } else {
                arr = Nd4j.create(minValues, countLength(list.get(0)));
            }
        } else if (details.oneHot) {
            arr = Nd4j.zeros(minValues, details.oneHotNumClasses);
        } else {
            if (details.subsetStart == details.subsetEndInclusive
                            && list.get(0).get(details.subsetStart) instanceof NDArrayWritable) {
                //Special case: single NDArrayWritable (example: ImageRecordReader)
                INDArray temp = ((NDArrayWritable) list.get(0).get(details.subsetStart)).get();
                val shape = ArrayUtils.clone(temp.shape());
                shape[0] = minValues;
                arr = Nd4j.create(shape);
            } else {
                //Need to check for multiple NDArrayWritables, or mixed NDArrayWritable + DoubleWritable etc
                int length = countLength(list.get(0), details.subsetStart, details.subsetEndInclusive);
                arr = Nd4j.create(minValues, length);
            }
        }

        for (int i = 0; i < minValues; i++) {
            List<Writable> c = list.get(i);
            if (details.entireReader) {
                //Convert entire reader contents, without modification
                INDArray converted = RecordConverter.toArray(c);
                putExample(arr, converted, i);
            } else if (details.oneHot) {
                //Convert a single column to a one-hot representation
                Writable w = c.get(details.subsetStart);
                //Index of class
                int classIdx = w.toInt();
                if (classIdx >= details.oneHotNumClasses) {
                    throw new IllegalStateException("Cannot convert sequence writables to one-hot: class index " + classIdx
                                    + " >= numClass (" + details.oneHotNumClasses + "). (Note that classes are zero-" +
                            "indexed, thus only values 0 to nClasses-1 are valid)");
                }
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

                        if (w instanceof NDArrayWritable) {
                            INDArray toPut = ((NDArrayWritable) w).get();
                            arr.put(new INDArrayIndex[] {NDArrayIndex.point(i),
                                            NDArrayIndex.interval(k, k + toPut.length())}, toPut);
                            k += toPut.length();
                        } else {
                            arr.putScalar(i, k, w.toDouble());
                            k++;
                        }
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
                    int maxTSLength, SubsetDetails details, int[] longestSequence, long rngSeed) {
        if (maxTSLength == -1)
            maxTSLength = list.get(0).size();
        INDArray arr;

        if (list.get(0).isEmpty()) {
            throw new ZeroLengthSequenceException("Zero length sequence encountered");
        }

        List<Writable> firstStep = list.get(0).get(0);

        int size = 0;
        if (details.entireReader) {
            //Need to account for NDArrayWritables etc in list:
            for (Writable w : firstStep) {
                if (w instanceof NDArrayWritable) {
                    size += ((NDArrayWritable) w).get().size(1);
                } else {
                    size++;
                }
            }
        } else if (details.oneHot) {
            size = details.oneHotNumClasses;
        } else {
            //Need to account for NDArrayWritables etc in list:
            for (int i = details.subsetStart; i <= details.subsetEndInclusive; i++) {
                Writable w = firstStep.get(i);
                if (w instanceof NDArrayWritable) {
                    size += ((NDArrayWritable) w).get().size(1);
                } else {
                    size++;
                }
            }
        }
        arr = Nd4j.create(new int[] {minValues, size, maxTSLength}, 'f');

        boolean needMaskArray = false;
        for (List<List<Writable>> c : list) {
            if (c.size() < maxTSLength)
                needMaskArray = true;
        }

        if (needMaskArray && alignmentMode == AlignmentMode.EQUAL_LENGTH) {
            throw new UnsupportedOperationException(
                            "Alignment mode is set to EQUAL_LENGTH but variable length data was "
                                            + "encountered. Use AlignmentMode.ALIGN_START or AlignmentMode.ALIGN_END with variable length data");
        }

        INDArray maskArray;
        if (needMaskArray) {
            maskArray = Nd4j.ones(minValues, maxTSLength);
        } else {
            maskArray = null;
        }

        //Don't use the global RNG as we need repeatability for each subset (i.e., features and labels must be aligned)
        Random rng = null;
        if (timeSeriesRandomOffset) {
            rng = new Random(rngSeed);
        }

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

            if (timeSeriesRandomOffset) {
                int maxPossible = maxTSLength - sequence.size() + 1;
                startOffset = rng.nextInt(maxPossible);
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

                        if (w instanceof NDArrayWritable) {
                            INDArray row = ((NDArrayWritable) w).get();

                            arr.put(new INDArrayIndex[] {NDArrayIndex.point(i),
                                            NDArrayIndex.interval(j, j + row.length()), NDArrayIndex.point(k)}, row);
                            j += row.length();
                        } else {
                            arr.putScalar(i, j, k, w.toDouble());
                            j++;
                        }
                    }
                } else if (details.oneHot) {
                    //Convert a single column to a one-hot representation
                    Writable w = null;
                    if (timeStep instanceof List)
                        w = timeStep.get(details.subsetStart);
                    else {
                        Iterator<Writable> iter = timeStep.iterator();
                        for (int x = 0; x <= details.subsetStart; x++)
                            w = iter.next();
                    }
                    int classIdx = w.toInt();
                    if (classIdx >= details.oneHotNumClasses) {
                        throw new IllegalStateException("Cannot convert sequence writables to one-hot: class index " + classIdx
                                        + " >= numClass (" + details.oneHotNumClasses + "). (Note that classes are zero-" +
                                "indexed, thus only values 0 to nClasses-1 are valid)");
                    }
                    arr.putScalar(i, classIdx, k, 1.0);
                } else {
                    //Convert a subset of the columns...
                    int l = 0;
                    for (int j = details.subsetStart; j <= details.subsetEndInclusive; j++) {
                        Writable w = timeStep.get(j);

                        if (w instanceof NDArrayWritable) {
                            INDArray row = ((NDArrayWritable) w).get();
                            arr.put(new INDArrayIndex[] {NDArrayIndex.point(i),
                                            NDArrayIndex.interval(l, l + row.length()), NDArrayIndex.point(k)}, row);

                            l += row.length();
                        } else {
                            arr.putScalar(i, l++, k, w.toDouble());
                        }
                    }
                }
            }

            //For any remaining time steps: set mask array to 0 (just padding)
            if (needMaskArray) {
                //Masking array entries at start (for align end)
                if (timeSeriesRandomOffset || alignmentMode == AlignmentMode.ALIGN_END) {
                    for (int t2 = 0; t2 < startOffset; t2++) {
                        maskArray.putScalar(i, t2, 0.0);
                    }
                }

                //Masking array entries at end (for align start)
                int lastStep = startOffset + sequence.size();
                if (timeSeriesRandomOffset || alignmentMode == AlignmentMode.ALIGN_START || lastStep < maxTSLength) {
                    for (int t2 = lastStep; t2 < maxTSLength; t2++) {
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
        return resetSupported;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        if(!resetSupported){
            throw new IllegalStateException("Cannot reset iterator - reset not supported (resetSupported() == false):" +
                    " one or more underlying (sequence) record readers do not support resetting");
        }

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

        private boolean timeSeriesRandomOffset = false;
        private long timeSeriesRandomOffsetSeed = System.currentTimeMillis();

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
            inputs.add(new SubsetDetails(readerName, false, true, numClasses, column, column));
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
            outputs.add(new SubsetDetails(readerName, false, true, numClasses, column, column));
            return this;
        }

        /**
         * For use with timeseries trained with tbptt
         * In a given minbatch, shorter time series are padded and appropriately masked to be the same length as the longest time series.
         * Cases with a skewed distrbution of lengths can result in the last few updates from the time series coming from mostly masked time steps.
         * timeSeriesRandomOffset randomly offsettsthe time series + masking appropriately to address this
         * @param timeSeriesRandomOffset, "true" to randomly offset time series within a minibatch
         * @param rngSeed seed for reproducibility
         */
        public Builder timeSeriesRandomOffset(boolean timeSeriesRandomOffset, long rngSeed) {
            this.timeSeriesRandomOffset = timeSeriesRandomOffset;
            this.timeSeriesRandomOffsetSeed = rngSeed;
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

        return nextMultiDataSet(nextRRVals, null, nextSeqRRVals, nextMetas);

    }

    @AllArgsConstructor
    private static class SubsetDetails implements Serializable {
        private final String readerName;
        private final boolean entireReader;
        private final boolean oneHot;
        private final int oneHotNumClasses;
        private final int subsetStart;
        private final int subsetEndInclusive;
    }
}
