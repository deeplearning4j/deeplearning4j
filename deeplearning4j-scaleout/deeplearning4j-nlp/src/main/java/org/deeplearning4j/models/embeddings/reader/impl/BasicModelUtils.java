package org.deeplearning4j.models.embeddings.reader.impl;

import com.google.common.collect.Lists;
import lombok.NonNull;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.SetUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Basic implementation for ModelUtils interface, suited for standalone use.
 *
 * PLEASE NOTE: This reader applies normalization to underlying lookup table.
 *
 * @author Adam Gibson
 */
public class BasicModelUtils<T extends SequenceElement> implements ModelUtils<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;

    protected boolean normalized  = false;

    private static final Logger log = LoggerFactory.getLogger(BasicModelUtils.class);

    public BasicModelUtils() {

    }

    @Override
    public void init(@NonNull WeightLookupTable<T> lookupTable) {
        this.vocabCache = lookupTable.getVocabCache();
        this.lookupTable = lookupTable;

        // reset normalization trigger on init call
        this.normalized = false;
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

        INDArray vec1 = lookupTable.vector(label1);
        INDArray vec2 = lookupTable.vector(label2);

        if (vec1 == null || vec2 == null) return Double.NaN;

        if (label1.equals(label2)) return 1.0;

        return Transforms.cosineSim(vec1, vec2);
    }


    @Override
    public Collection<String> wordsNearest(String label, int n) {
        Collection<String> collection = wordsNearest(Arrays.asList(label),new ArrayList<String>(),n + 1);
        if (collection.contains(label)) collection.remove(label);
        return collection;
    }

    /**
     * Accuracy based on questions which are a space separated list of strings
     * where the first word is the query word, the next 2 words are negative,
     * and the last word is the predicted word to be nearest
     * @param questions the questions to ask
     * @return the accuracy based on these questions
     */
    @Override
    public Map<String, Double> accuracy(List<String> questions) {
        Map<String,Double> accuracy = new HashMap<>();
        Counter<String> right = new Counter<>();
        String analogyType = "";
        for(String s : questions) {
            if(s.startsWith(":")) {
                double correct = right.getCount("correct");
                double wrong = right.getCount("wrong");
                if(analogyType.isEmpty()){
                    analogyType=s;
                    continue;
                }
                double accuracyRet = 100.0 * correct / (correct + wrong);
                accuracy.put(analogyType,accuracyRet);
                analogyType = s;
                right.clear();
            }
            else {
                String[] split = s.split(" ");
                String word = split[0];
                List<String> positive = Arrays.asList(word);
                List<String> negative = Arrays.asList(split[1],split[2]);
                String predicted = split[3];
                String w = wordsNearest(positive,negative,1).iterator().next();
                if(predicted.equals(w))
                    right.incrementCount("right",1.0);
                else
                    right.incrementCount("wrong",1.0);

            }
        }
        if(!analogyType.isEmpty()){
            double correct = right.getCount("correct");
            double wrong = right.getCount("wrong");
            double accuracyRet = 100.0 * correct / (correct + wrong);
            accuracy.put(analogyType,accuracyRet);
        }
        return accuracy;
    }

    /**
     * Find all words with a similar characters
     * in the vocab
     * @param word the word to compare
     * @param accuracy the accuracy: 0 to 1
     * @return the list of words that are similar in the vocab
     */
    @Override
    public List<String> similarWordsInVocabTo(String word, double accuracy) {
        List<String> ret = new ArrayList<>();
        for(String s : vocabCache.words()) {
            if(MathUtils.stringSimilarity(word, s) >= accuracy)
                ret.add(s);
        }
        return ret;
    }

    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        // Check every word is in the model
        for (String p : SetUtils.union(new HashSet<>(positive), new HashSet<>(negative))) {
            if (!vocabCache.containsWord(p)) {
                return new ArrayList<>();
            }
        }

        INDArray words = Nd4j.create(positive.size() + negative.size(), lookupTable.layerSize());
        int row = 0;
        //Set<String> union = SetUtils.union(new HashSet<>(positive), new HashSet<>(negative));
        for (String s : positive) {
            words.putRow(row++, lookupTable.vector(s));
        }

        for (String s : negative) {
            words.putRow(row++, lookupTable.vector(s).mul(-1));
        }

        INDArray mean = words.isMatrix() ? words.mean(0) : words;

        return wordsNearest(mean, top);
    }

    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    @Override
    public Collection<String> wordsNearestSum(String word, int n) {
        //INDArray vec = Transforms.unitVec(this.lookupTable.vector(word));
        INDArray vec = this.lookupTable.vector(word);
        return wordsNearestSum(vec, n);
    }

    /**
     * Words nearest based on positive and negative words
     * * @param top the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        if(lookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;

            INDArray syn0 = l.getSyn0();
            if (!normalized) {
                syn0.diviColumnVector(syn0.norm2(1));
                normalized = true;
            }

            INDArray similarity = Transforms.unitVec(words).mmul(syn0.transpose());
            // We assume that syn0 is normalized.
            // Hence, the following division is not needed anymore.
            // distances.diviRowVector(distances.norm2(1));
            //INDArray[] sorted = Nd4j.sortWithIndices(distances,0,false);
            List<Double> highToLowSimList = getTopN(similarity, top + 20);
            List<String> ret = new ArrayList<>();

            for (int i = 0; i < highToLowSimList.size(); i++) {
                String word = vocabCache.wordAtIndex(highToLowSimList.get(i).intValue());
                if (word != null && !word.equals("UNK") && !word.equals("STOP") ) {
                    ret.add(word);
                    if (ret.size() >= top) {
                        break;
                    }
                }
            }

            return ret;
        }

        Counter<String> distances = new Counter<>();

        for(String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(words, otherVec);
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
    private List<Double> getTopN(INDArray vec, int N) {
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

    /**
     * Words nearest based on positive and negative words
     * * @param top the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearestSum(INDArray words,int top) {

        if(lookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;
            INDArray syn0 = l.getSyn0();
            INDArray weights = syn0.norm2(0).rdivi(1).muli(words);
            INDArray distances = syn0.mulRowVector(weights).sum(1);
            INDArray[] sorted = Nd4j.sortWithIndices(distances,0,false);
            INDArray sort = sorted[0];
            List<String> ret = new ArrayList<>();
            if(top > sort.length())
                top = sort.length();
            //there will be a redundant word
            int end = top;
            for(int i = 0; i < end; i++) {
                String add = vocabCache.wordAtIndex(sort.getInt(i));
                if(add == null || add.equals("UNK") || add.equals("STOP")) {
                    end++;
                    if(end >= sort.length())
                        break;
                    continue;
                }
                ret.add(vocabCache.wordAtIndex(sort.getInt(i)));
            }
            return ret;
        }

        Counter<String> distances = new Counter<>();

        for(String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(words, otherVec);
            distances.incrementCount(s, sim);
        }

        distances.keepTopNKeys(top);
        return distances.keySet();
    }

    /**
     * Words nearest based on positive and negative words
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top) {
        INDArray words = Nd4j.create(lookupTable.layerSize());
    //    Set<String> union = SetUtils.union(new HashSet<>(positive), new HashSet<>(negative));
        for(String s : positive)
            words.addi(lookupTable.vector(s));


        for(String s : negative)
            words.addi(lookupTable.vector(s).mul(-1));

        return wordsNearestSum(words,top);
    }


    private static class ArrayComparator implements Comparator<Double[]> {
        @Override
        public int compare(Double[] o1, Double[] o2) {
            return Double.compare(o1[0], o2[0]);
        }
    }
}
