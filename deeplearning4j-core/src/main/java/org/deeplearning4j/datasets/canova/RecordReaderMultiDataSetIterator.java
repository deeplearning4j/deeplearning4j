package org.deeplearning4j.datasets.canova;

import lombok.AllArgsConstructor;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.*;

/**RecordReaderMultiDataSetIterator: An iterator for data from multiple RecordReaders and SequenceRecordReaders
 * The idea: generate multiple inputs and multiple outputs from one or more Sequence/RecordReaders. Inputs and outputs
 * may be obtained from subsets of the RecordReader and SequenceRecordReaders columns (for examples, some inputs and outputs
 * as different columns in the same file); it is also possible to mix different types of data (for example, using both
 * RecordReaders and SequenceRecordReaders in the same RecordReaderMultiDataSetIterator).
 */
public class RecordReaderMultiDataSetIterator implements MultiDataSetIterator {

    public enum AlignmentMode {
        EQUAL_LENGTH,
        ALIGN_START,
        ALIGN_END
    }

    private int batchSize;
    private AlignmentMode alignmentMode;
    private Map<String,RecordReader> recordReaders = new HashMap<>();
    private Map<String,SequenceRecordReader> sequenceRecordReaders = new HashMap<>();

    private List<SubsetDetails> inputs = new ArrayList<>();
    private List<SubsetDetails> outputs = new ArrayList<>();

    private MultiDataSetPreProcessor preProcessor;

    private RecordReaderMultiDataSetIterator(Builder builder){
        this.batchSize = builder.batchSize;
        this.alignmentMode = builder.alignmentMode;
        this.recordReaders = builder.recordReaders;
        this.sequenceRecordReaders = builder.sequenceRecordReaders;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public MultiDataSet next(int num) {
        if(!hasNext()) throw new NoSuchElementException("No next elements");

        //First: load the next values from the RR / SeqRRs
        Map<String,List<Collection<Writable>>> nextRRVals = new HashMap<>();
        Map<String,List<Collection<Collection<Writable>>>> nextSeqRRVals = new HashMap<>();

        int minValues = 0;
        for(Map.Entry<String,RecordReader> entry : recordReaders.entrySet() ){
            RecordReader rr = entry.getValue();
            List<Collection<Writable>> writables = new ArrayList<>(num);
            for( int i=0; i<num && rr.hasNext(); i++ ){
                writables.add(rr.next());
            }
            minValues = Math.min(minValues,writables.size());

            nextRRVals.put(entry.getKey(), writables);
        }

        for(Map.Entry<String,SequenceRecordReader> entry : sequenceRecordReaders.entrySet() ){
            SequenceRecordReader rr = entry.getValue();
            List<Collection<Collection<Writable>>> writables = new ArrayList<>(num);
            for( int i=0; i<num && rr.hasNext(); i++ ){
                writables.add(rr.sequenceRecord());
            }
            minValues = Math.min(minValues,writables.size());

            nextSeqRRVals.put(entry.getKey(), writables);
        }



        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public boolean hasNext() {
        for(RecordReader rr : recordReaders.values()) if(!rr.hasNext()) return false;
        for(SequenceRecordReader rr : sequenceRecordReaders.values()) if(!rr.hasNext()) return false;
        return true;
    }


    public static class Builder {

        private int batchSize;
        private AlignmentMode alignmentMode = AlignmentMode.EQUAL_LENGTH;
        private Map<String,RecordReader> recordReaders = new HashMap<>();
        private Map<String,SequenceRecordReader> sequenceRecordReaders = new HashMap<>();

        private List<SubsetDetails> inputs = new ArrayList<>();
        private List<SubsetDetails> outputs = new ArrayList<>();


        public Builder(int batchSize){
            this.batchSize = batchSize;
        }

        public void addReader(String readerName, RecordReader recordReader){
            recordReaders.put(readerName,recordReader);
        }

        public void addSequenceReader(String seqReaderName, SequenceRecordReader seqRecordReader){
            sequenceRecordReaders.put(seqReaderName,seqRecordReader);
        }

        /** Set the sequence alignment mode for all sequences */
        public void setSequenceAlignmentMode(AlignmentMode alignmentMode){
            this.alignmentMode = alignmentMode;
        }

        /** Set as an input, the entire contents (all columns) of the RecordReader or SequenceRecordReader */
        public void addInput(String readerName){
            inputs.add(new SubsetDetails(readerName,true,false,-1,-1,-1));
        }

        /** Set as an input, a subset of the specified RecordReader or SequenceRecordReader
         * @param readerName Name of the reader
         * @param columnFirst First column index, inclusive
         * @param columnLast Last column index, inclusive
         */
        public void addInput(String readerName, int columnFirst, int columnLast ){
            inputs.add(new SubsetDetails(readerName,false,false,-1,columnFirst,columnLast));
        }

        /** Add as an input a single column from the specified RecordReader / SequenceRecordReader
         * The assumption is that the specified column contains integer values in range 0..numClasses-1;
         * this integer will be converted to a one-hot representation
         * @param readerName
         * @param column
         * @param numClasses
         */
        public void addInputOneHot(String readerName, int column, int numClasses){
            inputs.add(new SubsetDetails(readerName,false,true,numClasses,column,-1));
        }

        /** Set as an output, the entire contents (all columns) of the RecordReader or SequenceRecordReader */
        public void addOutput(String readerName){
            outputs.add(new SubsetDetails(readerName,true,false,-1,-1,-1));
        }

        /**
         * @param readerName Name of the reader
         * @param columnFirst First column index
         * @param columnLast Last column index (inclusive)
         */
        public void addOutput(String readerName, int columnFirst, int columnLast){
            outputs.add(new SubsetDetails(readerName,false,false,-1,columnFirst,columnLast));
        }

        /** An an output, where the output is taken from a single column from the specified RecordReader / SequenceRecordReader
         * The assumption is that the specified column contains integer values in range 0..numClasses-1;
         * this integer will be converted to a one-hot representation (usually for classification)
         * @param readerName Name of the RecordReader / SequenceRecordReader
         * @param column index of the column
         * @param numClasses Number of classes
         */
        public void addOutputOneHot(String readerName, int column, int numClasses ){
            outputs.add(new SubsetDetails(readerName,false,true,numClasses,column,-1));
        }

        public RecordReaderMultiDataSetIterator build(){
            return new RecordReaderMultiDataSetIterator(this);
        }
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
