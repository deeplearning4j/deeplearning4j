package org.deeplearning4j.iterator.provider;

import lombok.NonNull;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.MathUtils;

import java.util.*;

/**
 * Iterate over a set of sentences/documents,
 * where the sentences and labels are provided in lists.
 *
 * @author Alex Black
 */
public class CollectionLabeledSentenceProvider implements LabeledSentenceProvider {

    private final List<String> sentences;
    private final List<String> labels;
    private final Random rng;
    private final int[] order;
    private final List<String> allLabels;

    private int cursor = 0;

    public CollectionLabeledSentenceProvider(@NonNull List<String> sentences,
                    @NonNull List<String> labelsForSentences) {
        this(sentences, labelsForSentences, new Random());
    }

    public CollectionLabeledSentenceProvider(@NonNull List<String> sentences, @NonNull List<String> labelsForSentences,
                    Random rng) {
        if (sentences.size() != labelsForSentences.size()) {
            throw new IllegalArgumentException("Sentences and labels must be same size (sentences size: "
                            + sentences.size() + ", labels size: " + labelsForSentences.size() + ")");
        }

        this.sentences = sentences;
        this.labels = labelsForSentences;
        this.rng = rng;
        if (rng == null) {
            order = null;
        } else {
            order = new int[sentences.size()];
            for (int i = 0; i < sentences.size(); i++) {
                order[i] = i;
            }

            MathUtils.shuffleArray(order, rng);
        }

        //Collect set of unique labels for all sentences
        Set<String> uniqueLabels = new HashSet<>();
        for (String s : labelsForSentences) {
            uniqueLabels.add(s);
        }
        allLabels = new ArrayList<>(uniqueLabels);
        Collections.sort(allLabels);
    }

    @Override
    public boolean hasNext() {
        return cursor < sentences.size();
    }

    @Override
    public Pair<String, String> nextSentence() {
        int idx;
        if (rng == null) {
            idx = cursor++;
        } else {
            idx = order[cursor++];
        }
        return new Pair<>(sentences.get(idx), labels.get(idx));
    }

    @Override
    public void reset() {
        cursor = 0;
        if (rng != null) {
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public int totalNumSentences() {
        return sentences.size();
    }

    @Override
    public List<String> allLabels() {
        return allLabels;
    }

    @Override
    public int numLabelClasses() {
        return allLabels.size();
    }
}
