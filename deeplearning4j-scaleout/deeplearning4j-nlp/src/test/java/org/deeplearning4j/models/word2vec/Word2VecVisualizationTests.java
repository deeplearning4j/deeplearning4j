package org.deeplearning4j.models.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

/**
 * This set of tests will address TSNE building checks, as well as parts of UI package involved there
 *
 *
 * @author raver119@gmail.com
 */
@Ignore
public class Word2VecVisualizationTests {

    private static WordVectors vectors;

    @Before
    public synchronized void setUp() throws Exception {
        if (vectors == null) {
            vectors = WordVectorSerializer.loadFullModel("/ext/Temp/Models/raw_sentences.dat");
        }
    }

    @Test
    public void testBarnesHutTsneVisualization() throws Exception {
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(4)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();

        //vectors.lookupTable().plotVocab(tsne);


    }
}
