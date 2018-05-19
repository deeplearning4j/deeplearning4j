package org.deeplearning4j.models.embeddings.reader.impl;

import com.google.common.collect.Lists;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.util.SetUtils;

import java.util.*;

/**
 * Basic implementation for ModelUtils interface, suited for standalone use.
 *
 * PLEASE NOTE: This reader applies normalization to underlying lookup table.
 *
 * @author Adam Gibson
 */
@Slf4j
public class BasicModelUtils<T extends SequenceElement> implements ModelUtils<T> {
    public static final String EXISTS = "exists";
    public static final String CORRECT = "correct";
    public static final String WRONG = "wrong";
    protected volatile VocabCache<T> vocabCache;
    protected volatile WeightLookupTable<T> lookupTable;

    protected volatile boolean normalized = false;


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
    public double similarity(@NonNull String label1, @NonNull String label2) {
        if (label1 == null || label2 == null) {
            log.debug("LABELS: " + label1 + ": " + (label1 == null ? "null" : EXISTS) + ";" + label2 + " vec2:"
                            + (label2 == null ? "null" : EXISTS));
            return Double.NaN;
        }

        if (!vocabCache.hasToken(label1)) {
            log.debug("Unknown token 1 requested: [{}]", label1);
            return Double.NaN;
        }

        if (!vocabCache.hasToken(label2)) {
            log.debug("Unknown token 2 requested: [{}]", label2);
            return Double.NaN;
        }

        INDArray vec1 = lookupTable.vector(label1).dup();
        INDArray vec2 = lookupTable.vector(label2).dup();


        if (vec1 == null || vec2 == null) {
            log.debug(label1 + ": " + (vec1 == null ? "null" : EXISTS) + ";" + label2 + " vec2:"
                            + (vec2 == null ? "null" : EXISTS));
            return Double.NaN;
        }

        if (label1.equals(label2))
            return 1.0;

        return Transforms.cosineSim(vec1, vec2);
    }


    @Override
    public Collection<String> wordsNearest(String label, int n) {
        List<String> collection = new ArrayList<>(wordsNearest(Arrays.asList(label), new ArrayList<String>(), n + 1));
        if (collection.contains(label))
            collection.remove(label);

        while (collection.size() > n)
            collection.remove(collection.size() - 1);

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
        Map<String, Double> accuracy = new HashMap<>();
        Counter<String> right = new Counter<>();
        String analogyType = "";
        for (String s : questions) {
            if (s.startsWith(":")) {
                double correct = right.getCount(CORRECT);
                double wrong = right.getCount(WRONG);
                if (analogyType.isEmpty()) {
                    analogyType = s;
                    continue;
                }
                double accuracyRet = 100.0 * correct / (correct + wrong);
                accuracy.put(analogyType, accuracyRet);
                analogyType = s;
                right.clear();
            } else {
                String[] split = s.split(" ");
                String word = split[0];
                List<String> positive = Arrays.asList(word);
                List<String> negative = Arrays.asList(split[1], split[2]);
                String predicted = split[3];
                String w = wordsNearest(positive, negative, 1).iterator().next();
                if (predicted.equals(w))
                    right.incrementCount(CORRECT, 1.0f);
                else
                    right.incrementCount(WRONG, 1.0f);

            }
        }
        if (!analogyType.isEmpty()) {
            double correct = right.getCount(CORRECT);
            double wrong = right.getCount(WRONG);
            double accuracyRet = 100.0 * correct / (correct + wrong);
            accuracy.put(analogyType, accuracyRet);
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
        for (String s : vocabCache.words()) {
            if (MathUtils.stringSimilarity(word, s) >= accuracy)
                ret.add(s);
        }
        return ret;
    }

    public Collection<String> wordsNearest(@NonNull Collection<String> positive, @NonNull Collection<String> negative,
                    int top) {
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

        Collection<String> tempRes = wordsNearest(mean, top + positive.size() + negative.size());
        List<String> realResults = new ArrayList<>();

        for (String word : tempRes) {
            if (!positive.contains(word) && !negative.contains(word) && realResults.size() < top)
                realResults.add(word);
        }

        return realResults;
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
        if (lookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;

            INDArray syn0 = l.getSyn0();

            if (!normalized) {
                synchronized (this) {
                    if (!normalized) {
                        syn0.diviColumnVector(syn0.norm2(1));
                        normalized = true;
                    }
                }
            }

            INDArray similarity = Transforms.unitVec(words).mmul(syn0.transpose());

            List<Double> highToLowSimList = getTopN(similarity, top + 20);

            List<WordSimilarity> result = new ArrayList<>();

            for (int i = 0; i < highToLowSimList.size(); i++) {
                String word = vocabCache.wordAtIndex(highToLowSimList.get(i).intValue());
                if (word != null && !word.equals("UNK") && !word.equals("STOP")) {
                    INDArray otherVec = lookupTable.vector(word);
                    double sim = Transforms.cosineSim(words, otherVec);

                    result.add(new WordSimilarity(word, sim));
                }
            }

            Collections.sort(result, new SimilarityComparator());

            return getLabels(result, top);
        }

        Counter<String> distances = new Counter<>();

        for (String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(words, otherVec);
            distances.incrementCount(s, (float) sim);
        }


        distances.keepTopNElements(top);
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
        PriorityQueue<Double[]> queue = new PriorityQueue<>(vec.rows(), comparator);

        for (int j = 0; j < vec.length(); j++) {
            final Double[] pair = new Double[] {vec.getDouble(j), (double) j};
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
    public Collection<String> wordsNearestSum(INDArray words, int top) {

        if (lookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;
            INDArray syn0 = l.getSyn0();
            INDArray weights = syn0.norm2(0).rdivi(1).muli(words);
            INDArray distances = syn0.mulRowVector(weights).sum(1);
            INDArray[] sorted = Nd4j.sortWithIndices(distances, 0, false);
            INDArray sort = sorted[0];
            List<String> ret = new ArrayList<>();

            // FIXME: int cast
            if (top > sort.length())
                top = (int) sort.length();
            //there will be a redundant word
            int end = top;
            for (int i = 0; i < end; i++) {
                String add = vocabCache.wordAtIndex(sort.getInt(i));
                if (add == null || add.equals("UNK") || add.equals("STOP")) {
                    end++;
                    if (end >= sort.length())
                        break;
                    continue;
                }
                ret.add(vocabCache.wordAtIndex(sort.getInt(i)));
            }
            return ret;
        }

        Counter<String> distances = new Counter<>();

        for (String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(words, otherVec);
            distances.incrementCount(s, (float) sim);
        }

        distances.keepTopNElements(top);
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
        for (String s : positive)
            words.addi(lookupTable.vector(s));


        for (String s : negative)
            words.addi(lookupTable.vector(s).mul(-1));

        return wordsNearestSum(words, top);
    }


    public static class SimilarityComparator implements Comparator<WordSimilarity> {
        @Override
        public int compare(WordSimilarity o1, WordSimilarity o2) {
            if (Double.isNaN(o1.getSimilarity()) && Double.isNaN(o2.getSimilarity())) {
                return 0;
            } else if (Double.isNaN(o1.getSimilarity()) && !Double.isNaN(o2.getSimilarity())) {
                return -1;
            } else if (!Double.isNaN(o1.getSimilarity()) && Double.isNaN(o2.getSimilarity())) {
                return 1;
            }
            return Double.compare(o2.getSimilarity(), o1.getSimilarity());
        }
    }

    public static class ArrayComparator implements Comparator<Double[]> {
        @Override
        public int compare(Double[] o1, Double[] o2) {
            if (Double.isNaN(o1[0]) && Double.isNaN(o2[0])) {
                return 0;
            } else if (Double.isNaN(o1[0]) && !Double.isNaN(o2[0])) {
                return -1;
            } else if (!Double.isNaN(o1[0]) && Double.isNaN(o2[0])) {
                return 1;
            }
            return Double.compare(o1[0], o2[0]);
        }
    }

    @Data
    @AllArgsConstructor
    public static class WordSimilarity {
        private String word;
        private double similarity;
    }

    public static List<String> getLabels(List<WordSimilarity> results, int limit) {
        List<String> result = new ArrayList<>();
        for (int x = 0; x < results.size(); x++) {
            result.add(results.get(x).getWord());
            if (result.size() >= limit)
                break;
        }

        return result;
    }
}
