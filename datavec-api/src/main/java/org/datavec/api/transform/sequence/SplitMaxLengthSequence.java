package org.datavec.api.transform.sequence;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**Split a sequence into a number of smaller sequences of length 'maxSequenceLength'.
 * If the sequence length is smaller than maxSequenceLength, the sequence is unchanged
 * Created by Alex on 16/03/2016.
 */
public class SplitMaxLengthSequence implements SequenceSplit {

    private final int maxSequenceLength;
    private final boolean equalSplits;
    private Schema inputSchema;

    /**
     * @param maxSequenceLength    max length of sequences
     * @param equalSplits   if true: split larger sequences inte equal sized subsequences. If false: split into
     */
    public SplitMaxLengthSequence(int maxSequenceLength, boolean equalSplits){
        this.maxSequenceLength = maxSequenceLength;
        this.equalSplits = equalSplits;
    }

    public List<List<List<Writable>>> split(List<List<Writable>> sequence) {
        int n = sequence.size();
        if(n <= maxSequenceLength) return Collections.singletonList(sequence);
        int splitSize;
        if(equalSplits){
            if(n % maxSequenceLength == 0){
                splitSize = n / maxSequenceLength;
            } else {
                splitSize = n / maxSequenceLength + 1;
            }
        } else {
            splitSize = maxSequenceLength;
        }

        List<List<List<Writable>>> out = new ArrayList<>();
        List<List<Writable>> current = new ArrayList<>(splitSize);
        for(List<Writable> step : sequence ){
            if(current.size() >= splitSize ){
                out.add(current);
                current = new ArrayList<>(splitSize);
            }
            current.add(step);
        }
        out.add(current);

        return out;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }
}
