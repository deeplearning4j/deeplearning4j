/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.word2vec;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectorsTest;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.documentiterator.*;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;


@Slf4j
@Tag(TagNames.FILE_IO)
@NativeTag
public class Word2VecTestsSmall extends BaseDL4JTest {
    WordVectors word2vec;

    @Override
    public long getTimeoutMilliseconds() {
        return isIntegrationTests() ? 240000 : 60000;
    }

    @BeforeEach
    public void setUp() throws Exception {
        word2vec = WordVectorSerializer.readWord2VecModel(new ClassPathResource("vec.bin").getFile());
    }

    @Test
    public void testWordsNearest2VecTxt() {
        String word = "Adam";
        String expectedNeighbour = "is";
        int neighbours = 1;

        Collection<String> nearestWords = word2vec.wordsNearest(word, neighbours);
        System.out.println(nearestWords);
        assertEquals(expectedNeighbour, nearestWords.iterator().next());
    }

    @Test
    public void testWordsNearest2NNeighbours() {
        String word = "Adam";
        int neighbours = 2;

        Collection<String> nearestWords = word2vec.wordsNearest(word, neighbours);
        System.out.println(nearestWords);
        assertEquals(neighbours, nearestWords.size());
    }

    @Test()
    @Timeout(300000)
    public void testUnkSerialization_1() throws Exception {
        val inputFile = Resources.asFile("big/raw_sentences.txt");
//        val iter = new BasicLineIterator(inputFile);
        SentenceIterator iter = ParagraphVectorsTest.getIterator(isIntegrationTests(), inputFile);
        val t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        val vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .epochs(1)
                .layerSize(300)
                .limitVocabularySize(1) // Limit the vocab size to 2 words
                .windowSize(5)
                .allowParallelTokenization(true)
                .batchSize(512)
                .learningRate(0.025)
                .minLearningRate(0.0001)
                .negativeSample(0.0)
                .sampling(0.0)
                .useAdaGrad(false)
                .useHierarchicSoftmax(true)
                .iterations(1)
                .useUnknown(true) // Using UNK with limited vocab size causes the issue
                .seed(42)
                .iterate(iter)
                .workers(4)
                .tokenizerFactory(t).build();

        vec.fit();

        val tmpFile = File.createTempFile("temp","temp");
        tmpFile.deleteOnExit();

        WordVectorSerializer.writeWord2VecModel(vec, tmpFile); // NullPointerException was thrown here
    }



    @Test
    public void testShardedLabelAwareIterator() {
        // Create a dummy LabelAwareIterator with sample documents
        LabelAwareIterator dummyIterator = new DummyLabelAwareIterator();

        // Create a tokenizer factory for the ShardedLabelAwareIterator
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        // Instantiate the ShardedLabelAwareIterator with a document size limit of 3 tokens
        ShardedLabelAwareIterator shardedIterator = new ShardedLabelAwareIterator(dummyIterator, tokenizerFactory, 3);

        // Store expected documents after sharding
        // Store expected documents after sharding
        List<String> expectedDocuments = Arrays.asList(
                "This is a",
                "sample document with",
                "some text for",
                "testing purposes",
                "Here is another",
                "document"
        );
        // Iterate through the sharded documents and check if they match the expected documents
        List<String> shardedDocuments = new ArrayList<>();
        while (shardedIterator.hasNext()) {
            LabelledDocument document = shardedIterator.next();
            shardedDocuments.add(document.getContent());
        }

        assertEquals(expectedDocuments, shardedDocuments);

        // Test reset functionality
        shardedIterator.reset();
        assertFalse(shardedIterator.getDocBatches() != null && !shardedIterator.getDocBatches().isEmpty());
    }

    // A simple dummy LabelAwareIterator implementation for testing purposes
    private static class DummyLabelAwareIterator implements LabelAwareIterator {
        private List<LabelledDocument> documents;
        private int currentIndex;

        public DummyLabelAwareIterator() {
            documents = new ArrayList<>();
            LabelledDocument doc1 = new LabelledDocument();
            doc1.setContent("This is a sample document with some text for testing purposes");
            documents.add(doc1);

            LabelledDocument doc2 = new LabelledDocument();
            doc2.setContent("Here is another document");
            documents.add(doc2);

            currentIndex = 0;
        }

        @Override
        public boolean hasNextDocument() {
            return currentIndex < documents.size();
        }

        @Override
        public LabelledDocument nextDocument() {
            return hasNextDocument() ? documents.get(currentIndex++) : null;
        }

        @Override
        public void reset() {
            currentIndex = 0;
        }

        @Override
        public LabelsSource getLabelsSource() {
            return new LabelsSource();
        }

        @Override
        public void shutdown() {
        }

        @Override
        public boolean hasNext() {
            return hasNextDocument();
        }

        @Override
        public LabelledDocument next() {
            return nextDocument();
        }
    }

    @Test
    public void testLabelAwareIterator_1() throws Exception {
        val resource = new ClassPathResource("/labeled");
        val file = resource.getFile();

        val iter = (LabelAwareIterator) new FileLabelAwareIterator.Builder().addSourceFolder(file).build();

        val t = new DefaultTokenizerFactory();

        val w2v = new Word2Vec.Builder()
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        // we hope nothing is going to happen here
    }

    @Test
    public void testPlot() {
        //word2vec.lookupTable().plotVocab();
    }


    @Test()
    @Timeout(300000)
    public void testW2VEmbeddingLayerInit() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);

        val inputFile = Resources.asFile("big/raw_sentences.txt");
        val iter = ParagraphVectorsTest.getIterator(isIntegrationTests(), inputFile);
//        val iter = new BasicLineIterator(inputFile);
        val t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .epochs(1)
                .layerSize(300)
                .limitVocabularySize(1) // Limit the vocab size to 2 words
                .windowSize(5)
                .allowParallelTokenization(true)
                .batchSize(512)
                .learningRate(0.025)
                .minLearningRate(0.0001)
                .negativeSample(0.0)
                .sampling(0.0)
                .useAdaGrad(false)
                .useHierarchicSoftmax(true)
                .iterations(1)
                .useUnknown(true) // Using UNK with limited vocab size causes the issue
                .seed(42)
                .iterate( iter)
                .workers(4)
                .tokenizerFactory(t).build();

        vec.fit();

        INDArray w = vec.lookupTable().getWeights();
        System.out.println(w);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345).list()
                .layer(new EmbeddingLayer.Builder().weightInit(vec).build())
                .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(w.size(1)).nOut(3).build())
                .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(3)
                        .nOut(4).build())
                .build();

        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray w0 = net.getParam("0_W");
        assertEquals(w, w0);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(net, baos, true);
        byte[] bytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        final MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(bais, true);

        assertEquals(net.getLayerWiseConfigurations(), restored.getLayerWiseConfigurations());
        assertTrue(net.params().equalsWithEps(restored.params(), 2e-3));
    }
}
