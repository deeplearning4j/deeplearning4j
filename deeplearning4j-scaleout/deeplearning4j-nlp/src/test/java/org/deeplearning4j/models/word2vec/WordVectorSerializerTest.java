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

import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 9/21/14.
 */
public class WordVectorSerializerTest {

    private File textFile,binaryFile;

    @Before
    public void before() throws Exception {
        if(textFile == null) {
            textFile = new ClassPathResource("vec.txt").getFile();
        }
        if(binaryFile == null) {
            binaryFile = new ClassPathResource("vec.bin").getFile();
        }
        FileUtils.deleteDirectory(new File("word2vec-index"));

    }

    @Test
    public void testLoaderText() throws IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(textFile, false);
        assertEquals(5,vec.vocab().numWords());
        assertTrue(vec.vocab().numWords() > 0);
    }


    @Test
    public void testWriteWordVectors() throws IOException {
        String pathToFile = "test_writing_word_vector.txt";

        Word2Vec vec = WordVectorSerializer.loadGoogleModel(textFile, false);
        InMemoryLookupTable lookupTable = (InMemoryLookupTable) vec.lookupTable();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache) vec.vocab();
        WordVectorSerializer.writeWordVectors(lookupTable, lookupCache, pathToFile);

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(pathToFile));
        assertTrue(wordVectors.getWordVector("<s>").length == 1000);
        assertTrue(wordVectors.getWordVector("Adam").length == 1000);
        assertTrue(wordVectors.getWordVector("awesome").length == 1000);
    }

    @Test
    public void testLoaderBinary() throws  IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
        assertEquals(2, vec.vocab().numWords());
    }


    @Test
    public void testBinaryDryRun() throws Exception {
        double vector;
        int words, size;
        String url = "";

        String path = "GoogleNews-vectors-negative300.bin.gz";
        File modelFile = new File(path);
        if(!modelFile.exists()) {
            FileUtils.copyURLToFile(new URL(url),modelFile);
        }
        try (BufferedInputStream bis =
                     new BufferedInputStream(GzipUtils.isCompressedFilename(modelFile.getName()) ?
                             new GZIPInputStream(new FileInputStream(modelFile)) :
                             new FileInputStream(modelFile));
             DataInputStream dis = new DataInputStream(bis)) {
            words = Integer.parseInt(WordVectorSerializer.readString(dis));
            size = Integer.parseInt(WordVectorSerializer.readString(dis));
            int wordsLoaded = 0;
            String word;

            for (int i = 0; i < words; i++) {

                word = WordVectorSerializer.readString(dis);
                if (word.isEmpty()) {
                    continue;
                }

                for (int j = 0; j < size; j++) {
                    vector = readFloat(dis);
                    System.out.println(vector);
                }

                wordsLoaded++;
                System.out.println("Loaded " + word + " and num words " + wordsLoaded + " out of " + words);


            }

            assertEquals(3000000,wordsLoaded);

        }
    }


    public static float readFloat(InputStream is) throws IOException {
        byte[] bytes = new byte[4];
        is.read(bytes);
        return getFloat(bytes);
    }

    /**
     * 读取一个float
     *
     * @param b
     * @return
     */
    public static float getFloat(byte[] b) {
        int accum = 0;
        accum = accum | (b[0] & 0xff) << 0;
        accum = accum | (b[1] & 0xff) << 8;
        accum = accum | (b[2] & 0xff) << 16;
        accum = accum | (b[3] & 0xff) << 24;
        return Float.intBitsToFloat(accum);
    }

    @Test
    public void testCurrentFile() throws Exception {
        Nd4j.dtype = DataBuffer.Type.FLOAT;
        String url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz";
        String path = "GoogleNews-vectors-negative300.bin.gz";
        File toDl = new File(path);
        if(!toDl.exists()) {
            FileUtils.copyURLToFile(new URL(url),toDl);
        }
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(toDl, true);
        assertEquals(3000000,vec.vocab().numWords());

    }

    @Test
    public void testTsne() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        ClassPathResource resource = new ClassPathResource("words.txt");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .theta(0.5).learningRate(500).setMaxIter(2000).build();
        WordVectors vec = WordVectorSerializer.loadTxtVectors(resource.getFile());
        InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
        List<String> labels = new ArrayList<>(vec.vocab().words());
        tsne.plot(table.getSyn0().divRowVector(table.getSyn0().norm2(0)),2,labels);
    }



}
