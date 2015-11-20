/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.word2vec;

import com.google.common.primitives.Doubles;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class WordVectorSerializerTest {

    private File textFile, binaryFile, textFile2;
    String pathToWriteto;

    @Before
    public void before() throws Exception {
        if(textFile == null) {
            textFile = new ClassPathResource("word2vecserialization/google_news_30.txt").getFile();
        }
        if(binaryFile == null) {
            binaryFile = new ClassPathResource("word2vecserialization/google_news_30.bin.gz").getFile();
        }
        pathToWriteto =  new ClassPathResource("word2vecserialization/testing_word2vec_serialization.txt")
                .getFile().getAbsolutePath();
        FileUtils.deleteDirectory(new File("word2vec-index"));
    }

    @Test
    @Ignore
    public void testLoaderTextSmall() throws Exception {
        INDArray vec = Nd4j.create(new double[]{0.002001,0.002210,-0.001915,-0.001639,0.000683,0.001511,0.000470,0.000106,-0.001802,0.001109,-0.002178,0.000625,-0.000376,-0.000479,-0.001658,-0.000941,0.001290,0.001513,0.001485,0.000799,0.000772,-0.001901,-0.002048,0.002485,0.001901,0.001545,-0.000302,0.002008,-0.000247,0.000367,-0.000075,-0.001492,0.000656,-0.000669,-0.001913,0.002377,0.002190,-0.000548,-0.000113,0.000255,-0.001819,-0.002004,0.002277,0.000032,-0.001291,-0.001521,-0.001538,0.000848,0.000101,0.000666,-0.002107,-0.001904,-0.000065,0.000572,0.001275,-0.001585,0.002040,0.000463,0.000560,-0.000304,0.001493,-0.001144,-0.001049,0.001079,-0.000377,0.000515,0.000902,-0.002044,-0.000992,0.001457,0.002116,0.001966,-0.001523,-0.001054,-0.000455,0.001001,-0.001894,0.001499,0.001394,-0.000799,-0.000776,-0.001119,0.002114,0.001956,-0.000590,0.002107,0.002410,0.000908,0.002491,-0.001556,-0.000766,-0.001054,-0.001454,0.001407,0.000790,0.000212,-0.001097,0.000762,0.001530,0.000097,0.001140,-0.002476,0.002157,0.000240,-0.000916,-0.001042,-0.000374,-0.001468,-0.002185,-0.001419,0.002139,-0.000885,-0.001340,0.001159,-0.000852,0.002378,-0.000802,-0.002294,0.001358,-0.000037,-0.001744,0.000488,0.000721,-0.000241,0.000912,-0.001979,0.000441,0.000908,-0.001505,0.000071,-0.000030,-0.001200,-0.001416,-0.002347,0.000011,0.000076,0.000005,-0.001967,-0.002481,-0.002373,-0.002163,-0.000274,0.000696,0.000592,-0.001591,0.002499,-0.001006,-0.000637,-0.000702,0.002366,-0.001882,0.000581,-0.000668,0.001594,0.000020,0.002135,-0.001410,-0.001303,-0.002096,-0.001833,-0.001600,-0.001557,0.001222,-0.000933,0.001340,0.001845,0.000678,0.001475,0.001238,0.001170,-0.001775,-0.001717,-0.001828,-0.000066,0.002065,-0.001368,-0.001530,-0.002098,0.001653,-0.002089,-0.000290,0.001089,-0.002309,-0.002239,0.000721,0.001762,0.002132,0.001073,0.001581,-0.001564,-0.001820,0.001987,-0.001382,0.000877,0.000287,0.000895,-0.000591,0.000099,-0.000843,-0.000563});
        String w1 = "database";
        String w2 = "DBMS";
        WordVectors vecModel = WordVectorSerializer.loadGoogleModel(new ClassPathResource("word2vec/googleload/sample_vec.txt").getFile(), false, true);
        WordVectors vectorsBinary = WordVectorSerializer.loadGoogleModel(new ClassPathResource("word2vec/googleload/sample_vec.bin").getFile(),true,true);
        INDArray textWeights = vecModel.lookupTable().getWeights();
        INDArray binaryWeights = vectorsBinary.lookupTable().getWeights();
        Collection<String> nearest = vecModel.wordsNearest("database", 10);
        Collection<String> nearestBinary = vectorsBinary.wordsNearest("database", 10);
        System.out.println(nearestBinary);
        assertEquals(vecModel.similarity("DBMS","DBMS's"),vectorsBinary.similarity("DBMS", "DBMS's"),1e-1);

    }

    @Test
    @Ignore
    public void testLoaderText() throws IOException {
        WordVectors vec = WordVectorSerializer.loadGoogleModel(textFile, false);
        assertEquals(vec.vocab().numWords(), 30);
        assertTrue(vec.vocab().hasToken("Morgan_Freeman"));
        assertTrue(vec.vocab().hasToken("JA_Montalbano"));
    }

    @Test
    public void testLoaderBinary() throws IOException {
        WordVectors vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
        assertEquals(vec.vocab().numWords(), 30);
        assertTrue(vec.vocab().hasToken("Morgan_Freeman"));
        assertTrue(vec.vocab().hasToken("JA_Montalbano"));
        double[] wordVector1 = vec.getWordVector("Morgan_Freeman");
        double[] wordVector2 = vec.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }

    @Test
    @Ignore
    public void testWriteWordVectors() throws IOException {
        WordVectors vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
        InMemoryLookupTable lookupTable = (InMemoryLookupTable) vec.lookupTable();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache) vec.vocab();
        WordVectorSerializer.writeWordVectors(lookupTable, lookupCache, pathToWriteto);

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(pathToWriteto));
        double[] wordVector1 = wordVectors.getWordVector("Morgan_Freeman");
        double[] wordVector2 = wordVectors.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }

    @Test
    @Ignore
    public void testWriteWordVectorsFromWord2Vec() throws IOException {
        WordVectors vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
        WordVectorSerializer.writeWordVectors(vec, pathToWriteto);

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(pathToWriteto));
        INDArray wordVector1 = wordVectors.getWordVectorMatrix("Morgan_Freeman");
        INDArray wordVector2 = wordVectors.getWordVectorMatrix("JA_Montalbano");
        assertEquals(vec.getWordVectorMatrix("Morgan_Freeman"),wordVector1);
        assertEquals(vec.getWordVectorMatrix("JA_Montalbano"),wordVector2);
        assertTrue(wordVector1.length() == 300);
        assertTrue(wordVector2.length() == 300);
        assertEquals(wordVector1.getDouble(0), 0.044423, 1e-3);
        assertEquals(wordVector2.getDouble(0), 0.051964, 1e-3);
    }

    @Test
    @Ignore
    public void testFromTableAndVocab() throws IOException {

        WordVectors vec = WordVectorSerializer.loadGoogleModel(textFile, false);
        InMemoryLookupTable lookupTable = (InMemoryLookupTable) vec.lookupTable();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache) vec.vocab();

        WordVectors wordVectors = WordVectorSerializer.fromTableAndVocab(lookupTable, lookupCache);
        double[] wordVector1 = wordVectors.getWordVector("Morgan_Freeman");
        double[] wordVector2 = wordVectors.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }


}
