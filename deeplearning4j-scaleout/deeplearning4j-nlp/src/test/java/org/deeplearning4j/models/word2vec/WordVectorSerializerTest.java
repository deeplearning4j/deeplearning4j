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
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

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
    public void testLoaderText() throws IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(textFile, false);
        assertEquals(vec.vocab().numWords(), 30);
        assertTrue(vec.vocab().hasToken("Morgan_Freeman"));
        assertTrue(vec.vocab().hasToken("JA_Montalbano"));
    }

    @Test
    public void testLoaderBinary() throws IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
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
    public void testWriteWordVectors() throws IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
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
    public void testWriteWordVectorsFromWord2Vec() throws IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
        WordVectorSerializer.writeWordVectors(vec, pathToWriteto);

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(pathToWriteto));
        double[] wordVector1 = wordVectors.getWordVector("Morgan_Freeman");
        double[] wordVector2 = wordVectors.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }

    @Test
    public void testFromTableAndVocab() throws IOException{

        Word2Vec vec = WordVectorSerializer.loadGoogleModel(textFile, false);
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

//    @Test
//    public void testTsne() throws Exception {
//        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
//        ClassPathResource resource = new ClassPathResource("words.txt");
//        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
//                .theta(0.5).learningRate(500).setMaxIter(2).build();
//        WordVectors vec = WordVectorSerializer.loadTxtVectors(resource.getFile());
//        InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
//        List<String> labels = new ArrayList<>(vec.vocab().words());
//        tsne.plot(table.getSyn0().divRowVector(table.getSyn0().norm2(0)), 2, labels);
//    }

//    public static float readFloat(InputStream is) throws IOException {
//        byte[] bytes = new byte[4];
//        is.read(bytes);
//        return getFloat(bytes);
//    }
//    public static float getFloat(byte[] b) {
//        int accum = 0;
//        accum = accum | (b[0] & 0xff) << 0;
//        accum = accum | (b[1] & 0xff) << 8;
//        accum = accum | (b[2] & 0xff) << 16;
//        accum = accum | (b[3] & 0xff) << 24;
//        return Float.intBitsToFloat(accum);
//    }
//   Will take a long time to actually to load the whole model
//    @Test
//    public void testBinaryDryRun() throws Exception {
//        double vector;
//        int words, size;
//        String url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz";
//
//        String path = "GoogleNews-vectors-negative300.bin.gz";
//        File modelFile = new File(path);
//        if(!modelFile.exists()) {
//            FileUtils.copyURLToFile(new URL(url),modelFile);
//        }
//        try (BufferedInputStream bis =
//                     new BufferedInputStream(GzipUtils.isCompressedFilename(modelFile.getName()) ?
//                             new GZIPInputStream(new FileInputStream(modelFile)) :
//                             new FileInputStream(modelFile));
//             DataInputStream dis = new DataInputStream(bis)) {
//            words = Integer.parseInt(WordVectorSerializer.readString(dis));
//            size = Integer.parseInt(WordVectorSerializer.readString(dis));
//            int wordsLoaded = 0;
//            String word;
//
//            for (int i = 0; i < words; i++) {
//
//                word = WordVectorSerializer.readString(dis);
//                if (word.isEmpty()) {
//                    continue;
//                }
//
//                for (int j = 0; j < size; j++) {
//                    vector = readFloat(dis);
//                    System.out.println(vector);
//                }
//
//                wordsLoaded++;
//                System.out.println("Loaded " + word + " and num words " + wordsLoaded + " out of " + words);
//            }
//
//            assertEquals(wordsLoaded, 3000000);
//        }
//    }
//    @Test
//    public void testCurrentFile() throws Exception {
//        Nd4j.dtype = DataBuffer.Type.FLOAT;
//        String url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz";
//        String path = "GoogleNews-vectors-negative300.bin.gz";
//        File toDl = new File(path);
//        if(!toDl.exists()) {
//            FileUtils.copyURLToFile(new URL(url),toDl);
//        }
//        Word2Vec vec = WordVectorSerializer.loadGoogleModel(toDl, true);
//        assertEquals(3000000,vec.vocab().numWords());
//
//    }
}
