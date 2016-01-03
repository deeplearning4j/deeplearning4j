package org.deeplearning4j.models.embeddings.reader.impl;

import com.google.common.collect.Lists;
import lombok.NonNull;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelReader;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.util.SetUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * Basic implementation for ModelReader interface, suited for standalone use.
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public class BasicReader<T extends SequenceElement> implements ModelReader<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;

    public BasicReader() {

    }

    @Override
    public void init(@NonNull VocabCache<T> vocabCache,@NonNull WeightLookupTable<T> lookupTable) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
    }

    /**
     * Returns the similarity of 2 words. Result value will be in range [-1,1], where -1.0 is exact opposite similarity, i.e. NO similarity, and 1.0 is total match of two word vectors.
     * However, most of time you'll see values in range [0,1], but that's something depends of training corpus.
     *
     * Returns NaN if any of labels not exists in vocab, or any label is null
     *
     * @param label1 the first word
     * @param label2 the second word
     * @return a normalized similarity (cosine similarity)
     */
    @Override
    public double similarity( String label1, String label2) {
        if (label1 == null || label2 == null) return Double.NaN;

        if (label1.equals(label2)) return 1.0;

        INDArray vec1 = lookupTable.vector(label1);
        INDArray vec2 = lookupTable.vector(label2);

        if (vec1 == null || vec2 == null) return Double.NaN;

        return Transforms.cosineSim(vec1, vec2);
    }

    @Override
    public Collection<String> wordsNearest(String label, int n) {
        return wordsNearest(Arrays.asList(label),new ArrayList<String>(),n);
    }

    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        // Check every word is in the model
        for (String p : SetUtils.union(new HashSet<>(positive), new HashSet<>(negative))) {
            if (!vocabCache.containsWord(p)) {
                return new ArrayList<>();
            }
        }

        WeightLookupTable weightLookupTable = lookupTable;
        INDArray words = Nd4j.create(positive.size() + negative.size(), weightLookupTable.layerSize());
        int row = 0;
        Set<String> union = SetUtils.union(new HashSet<>(positive), new HashSet<>(negative));
        for (String s : positive) {
            words.putRow(row++, weightLookupTable.vector(s));
        }

        for (String s : negative) {
            words.putRow(row++, weightLookupTable.vector(s).mul(-1));
        }

        INDArray mean = words.isMatrix() ? words.mean(0) : words;
        // TODO this should probably be replaced with wordsNearest(mean, top)
        if (weightLookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) weightLookupTable;

            INDArray syn0 = l.getSyn0();
            syn0.diviRowVector(syn0.norm2(0));

            INDArray similarity = Transforms.unitVec(mean).mmul(syn0.transpose());
            // We assume that syn0 is normalized.
            // Hence, the following division is not needed anymore.
            // distances.diviRowVector(distances.norm2(1));
            //INDArray[] sorted = Nd4j.sortWithIndices(distances,0,false);
            List<Double> highToLowSimList = getTopN(similarity, top + union.size());
            List<String> ret = new ArrayList<>();

            for (int i = 0; i < highToLowSimList.size(); i++) {
                String word = vocabCache.wordAtIndex(highToLowSimList.get(i).intValue());
                if (word != null && !word.equals("UNK") && !word.equals("STOP") && !union.contains(word)) {
                    ret.add(word);
                    if (ret.size() >= top) {
                        break;
                    }
                }
            }

            return ret;
        }

        Counter<String> distances = new Counter<>();

        for (String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(mean, otherVec);
            distances.incrementCount(s, sim);
        }

        distances.keepTopNKeys(top);
        return distances.keySet();

    }

    /**
     * Get top N elements
     *
     * @param vec the vec to extract the top elements from
     * @param N the number of elements to extract
     * @return the indices and the sorted top N elements
     */
    private static List<Double> getTopN(INDArray vec, int N) {
        ArrayComparator comparator = new ArrayComparator();
        PriorityQueue<Double[]> queue = new PriorityQueue<>(vec.rows(),comparator);

        for (int j = 0; j < vec.length(); j++) {
            final Double[] pair = new Double[]{vec.getDouble(j), (double) j};
            if (queue.size() < N) {
                queue.add(pair);
            } else {
                Double[] head = queue.peek();
                if (comparator.compare(pair, head) > 0) {
                    queue.poll();
                    queue.add(pair);
                }
            }
        }

        List<Double> lowToHighSimLst = new ArrayList<>();

        while (!queue.isEmpty()) {
            double ind = queue.poll()[1];
            lowToHighSimLst.add(ind);
        }
        return Lists.reverse(lowToHighSimLst);
    }

    private static class ArrayComparator implements Comparator<Double[]> {
        @Override
        public int compare(Double[] o1, Double[] o2) {
            return Double.compare(o1[0], o2[0]);
        }
    }
}
