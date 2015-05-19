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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

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
    public void testLoaderBinary() throws  IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(binaryFile, true);
        assertEquals(2,vec.vocab().numWords());

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


}
