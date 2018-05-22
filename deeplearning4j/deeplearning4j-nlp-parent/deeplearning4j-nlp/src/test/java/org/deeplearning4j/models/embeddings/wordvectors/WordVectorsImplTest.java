package org.deeplearning4j.models.embeddings.wordvectors;

import com.google.common.collect.Lists;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.when;

public class WordVectorsImplTest {
    private VocabCache vocabCache;
    private WeightLookupTable weightLookupTable;
    private WordVectorsImpl<SequenceElement> wordVectors;

    @Before
    public void init() throws Exception {
        vocabCache = Mockito.mock(VocabCache.class);
        weightLookupTable = Mockito.mock(WeightLookupTable.class);
        wordVectors = new WordVectorsImpl<>();
    }

    @Test
    public void getWordVectors_HaveTwoWordsNotInVocabAndOneIn_ExpectAllNonWordsRemoved() {
        INDArray wordVector = Nd4j.create(1, 1);
        wordVector.putScalar(0, 5);
        when(vocabCache.indexOf("word")).thenReturn(0);
        when(vocabCache.containsWord("word")).thenReturn(true);
        when(weightLookupTable.getWeights()).thenReturn(wordVector);
        wordVectors.setVocab(vocabCache);
        wordVectors.setLookupTable(weightLookupTable);

        INDArray indArray = wordVectors.getWordVectors(Lists.newArrayList("word", "here", "is"));

        assertEquals(wordVector, indArray);
    }
}
