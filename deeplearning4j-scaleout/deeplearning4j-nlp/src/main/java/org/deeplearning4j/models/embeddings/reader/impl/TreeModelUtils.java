package org.deeplearning4j.models.embeddings.reader.impl;

import lombok.NonNull;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class TreeModelUtils<T extends SequenceElement> extends BasicModelUtils<T> {
    protected VPTree vpTree;

    @Override
    public void init(@NonNull WeightLookupTable<T> lookupTable) {
        super.init(lookupTable);
        vpTree = null;
    }

    /**
     * This method returns nearest words for target word, based on tree structure.
     * This method is recommended to use if you're going to call for nearest words multiple times.
     * VPTree will be built upon firt call to this method
     *
     * @param label label of element we're looking nearest words to
     * @param n number of nearest elements to return
     * @return
     */
    @Override
    public Collection<String> wordsNearest(String label, int n) {
        if (!vocabCache.hasToken(label)) return new ArrayList<>();

        // build new tree if it wasn't created before
        if (vpTree == null) {
            List<DataPoint> points = new ArrayList<>();
            for (String word: vocabCache.words()) {
                points.add(new DataPoint(vocabCache.indexOf(word), lookupTable.vector(word)));
            }
            vpTree = new VPTree(points);

        }
        List<DataPoint> add = new ArrayList<>();
        List<Double> distances = new ArrayList<>();

        // we need n+1 to address original datapoint removal
        vpTree.search(new DataPoint(0, lookupTable.vector(label)), n+1, add, distances );

        Collection<String> ret = new ArrayList<>();
        for (DataPoint e: add) {
            String word  = vocabCache.wordAtIndex(e.getIndex());
            if (!word.equals(label)) ret.add(word);
        }

        return ret;
    }
}
