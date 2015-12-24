package org.deeplearning4j.models.glove.count;

import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 25.12.15.
 */
public class BinaryCoOccurrenceReaderTest {

    private static final Logger log = LoggerFactory.getLogger(BinaryCoOccurrenceReaderTest.class);

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testHasMoreObjects() throws Exception {
        File tempFile = File.createTempFile("tmp", "tmp");
        tempFile.deleteOnExit();

        VocabCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>()
                .build();

        VocabWord word1 = new VocabWord(1.0, "human");
        VocabWord word2 = new VocabWord(2.0, "animal");
        VocabWord word3 = new VocabWord(3.0, "unknown");

        vocabCache.addToken(word1);
        vocabCache.addToken(word2);
        vocabCache.addToken(word3);

        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        huffman.applyIndexes(vocabCache);


        BinaryCoOccurrenceWriter<VocabWord> writer = new BinaryCoOccurrenceWriter<>(tempFile);

        CoOccurrenceWeight<VocabWord> object1 = new CoOccurrenceWeight<VocabWord>();
        object1.setElement1(word1);
        object1.setElement2(word2);
        object1.setWeight(3.14159265);

        writer.writeObject(object1);

        CoOccurrenceWeight<VocabWord> object2 = new CoOccurrenceWeight<VocabWord>();
        object2.setElement1(word2);
        object2.setElement2(word3);
        object2.setWeight(0.197);

        writer.writeObject(object2);

        writer.finish();

        BinaryCoOccurrenceReader<VocabWord> reader = new BinaryCoOccurrenceReader<>(tempFile, vocabCache);

        int cnt = 0;
        while (reader.hasMoreObjects()) {
            CoOccurrenceWeight<VocabWord> r1 = reader.nextObject();
            log.info("Object received: " + r1);
            assertNotEquals(null, r1);
            cnt++;
        }

        assertEquals(2, cnt);
    }
}