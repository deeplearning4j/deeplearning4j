package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.assertEquals;

import java.util.Collection;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Before;
import org.junit.Test;


public class Word2VecTestsSmall
{
    WordVectors word2vec;

    @Before
    public void setUp()
        throws Exception
    {
        word2vec = WordVectorSerializer.loadGoogleModel(
                new ClassPathResource("vec.bin").getFile(), true,true);
    }

    @Test
    public void testWordsNearest2VecTxt()
    {
        String word = "Adam";
        String expectedNeighbour = "is";
        int neighbours = 1;

        Collection<String> nearestWords = word2vec.wordsNearest(word, neighbours);
        System.out.println(nearestWords);
        assertEquals(expectedNeighbour, nearestWords.iterator().next());
    }

    @Test
    public void testWordsNearest2NNeighbours()
    {
        String word = "Adam";
        int neighbours = 2;

        Collection<String> nearestWords = word2vec.wordsNearest(word, neighbours);
        System.out.println(nearestWords);
        assertEquals(neighbours, nearestWords.size());

    }

    @Test
    public void testPlot()
    {
        //word2vec.lookupTable().plotVocab();
    }
}
