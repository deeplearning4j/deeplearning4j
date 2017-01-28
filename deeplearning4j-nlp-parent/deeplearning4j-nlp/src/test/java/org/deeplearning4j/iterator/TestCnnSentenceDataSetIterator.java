package org.deeplearning4j.iterator;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * Created by Alex on 28/01/2017.
 */
public class TestCnnSentenceDataSetIterator {

    @Test
    public void testSentenceIterator() throws Exception {


        WordVectors w2v = WordVectorSerializer.readWord2VecModel(new ClassPathResource("word2vec/googleload/sample_vec.bin").getFile());

        int vectorSize = w2v.lookupTable().layerSize();

        Collection<String> words = w2v.lookupTable().getVocabCache().words();

//        for(String s : words){
//            System.out.println(s);
//        }

        List<String> sentences = new ArrayList<>();
        //First word: all present
        sentences.add("these balance Database model");
        sentences.add("into same THISWORDDOESNTEXIST are");
        int maxLength = 4;
        List<String> s1 = Arrays.asList("these", "balance", "Database", "model");
        List<String> s2 = Arrays.asList("into", "same", "are");

        List<String> labelsForSentences = Arrays.asList("Positive", "Negative");

        INDArray expLabels = Nd4j.create(new double[][]{{0,1},{1,0}});  //Order of labels: alphabetic. Positive -> [0,1]

        boolean[] alongHeightVals = new boolean[]{true, false};

        for(boolean alongHeight : alongHeightVals){

            INDArray expectedFeatures;
            if(alongHeight){
                expectedFeatures = Nd4j.create(2, 1, maxLength, vectorSize);
            } else {
                expectedFeatures = Nd4j.create(2, 1, vectorSize, maxLength);
            }

            INDArray expectedFeatureMask = Nd4j.create(new double[][]{
                    {1,1,1,1},
                    {1,1,1,0}});

            for( int i=0; i<4; i++ ){
                if(alongHeight){
                    expectedFeatures.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                            .assign(w2v.getWordVectorMatrix(s1.get(i)));
                } else {
                    expectedFeatures.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i))
                            .assign(w2v.getWordVectorMatrix(s1.get(i)));
                }
            }

            for( int i=0; i<3; i++ ){
                if(alongHeight){
                    expectedFeatures.get(NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                            .assign(w2v.getWordVectorMatrix(s2.get(i)));
                } else {
                    expectedFeatures.get(NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i))
                            .assign(w2v.getWordVectorMatrix(s2.get(i)));
                }
            }


            LabeledSentenceProvider p = new CollectionLabeledSentenceProvider(sentences, labelsForSentences, null);
            DataSetIterator dsi = new CnnSentenceDataSetIterator.Builder()
                    .labelledSentenceProvider(p)
                    .wordVectors(w2v)
                    .maxSentenceLength(256)
                    .minibatchSize(32)
                    .sentencesAlongHeight(alongHeight)
                    .build();

//            System.out.println("alongHeight = " + alongHeight);
            DataSet ds = dsi.next();
            assertArrayEquals(expectedFeatures.shape(), ds.getFeatures().shape());
            assertEquals(expectedFeatures, ds.getFeatures());
            assertEquals(expLabels, ds.getLabels());
            assertEquals(expectedFeatureMask, ds.getFeaturesMaskArray());
            assertNull(ds.getLabelsMaskArray());
        }
    }
}
