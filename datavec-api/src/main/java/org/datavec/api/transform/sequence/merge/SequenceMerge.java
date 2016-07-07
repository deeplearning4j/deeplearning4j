package org.datavec.api.transform.sequence.merge;

import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Merge multiple sequences into one single sequence.
 * Requires a SequenceComparator to determine the final ordering
 *
 * @author Alex Black
 */
public class SequenceMerge implements Serializable {

    private final SequenceComparator comparator;

    public SequenceMerge(SequenceComparator comparator){
        this.comparator = comparator;
    }

    public List<List<Writable>> mergeSequences(List<List<List<Writable>>> multipleSequences){

        //Approach here: append all time steps, then sort

        List<List<Writable>> out = new ArrayList<>();
        for(List<List<Writable>> sequence : multipleSequences){
            out.addAll(sequence);
        }

        Collections.sort(out,comparator);

        return out;
    }
}
