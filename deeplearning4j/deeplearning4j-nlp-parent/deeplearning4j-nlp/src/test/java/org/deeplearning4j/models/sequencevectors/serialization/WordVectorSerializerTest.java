package org.deeplearning4j.models.sequencevectors.serialization;

import lombok.val;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class WordVectorSerializerTest {
    private AbstractCache<VocabWord> cache;

    @Before
    public void setUp() throws Exception {
        cache = new AbstractCache.Builder<VocabWord>().build();

        val words = new VocabWord[3];
        words[0] = new VocabWord(1.0, "word");
        words[1] = new VocabWord(2.0, "test");
        words[2] = new VocabWord(3.0, "tester");

        for (int i = 0; i < words.length; ++i) {
            cache.addToken(words[i]);
            cache.addWordToIndex(i, words[i].getLabel());
        }
    }

    @Test
    public void sequenceVectorsCorrect_WhenDeserialized() {
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration()).
                vocabCache(cache).
                build();
        SequenceVectors<VocabWord> deser = null;
        try {
            String json = WordVectorSerializer.writeSequenceVectors(vectors);
            deser = WordVectorSerializer.readSequenceVectors(json);
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }

        assertEquals(cache.totalWordOccurrences(),deser.vocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), deser.vocab().totalNumberOfDocs());
        assertEquals(cache.numWords(), deser.vocab().numWords());

        for (int i = 0; i < cache.words().size(); ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = deser.vocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }

    }
}
