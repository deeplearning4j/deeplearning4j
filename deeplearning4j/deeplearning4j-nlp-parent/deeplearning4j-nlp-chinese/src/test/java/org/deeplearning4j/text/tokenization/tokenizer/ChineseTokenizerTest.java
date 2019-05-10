/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.text.tokenization.tokenizer;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerFactory.ChineseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**@author wangfeng
 * @date June 3,2017
 * @Description
 *
 */
@Slf4j
public class ChineseTokenizerTest {

    private final String toTokenize = "青山绿水和伟大的科学家让世界更美好和平";
    private final String[] expect = {"青山绿水", "和", "伟大", "的", "科学家", "让", "世界", "更", "美好", "和平"};

    @Test
    public void testChineseTokenizer() {
        TokenizerFactory tokenizerFactory = new ChineseTokenizerFactory();
        Tokenizer tokenizer = tokenizerFactory.create(toTokenize);
        assertEquals(expect.length, tokenizer.countTokens());
        for (int i = 0; i < tokenizer.countTokens(); ++i) {
            assertEquals(tokenizer.nextToken(), expect[i]);
        }
    }

    //Train model by some data of the chinese names,Then find out the names from the dataset
    @Ignore
    @Test
    public void testFindNamesFromText() throws IOException {
        SentenceIterator iter = new BasicLineIterator("src/test/resources/chineseName.txt");

        log.info("load is right!");
        TokenizerFactory tokenizerFactory = new ChineseTokenizerFactory();
        //tokenizerFactory.setTokenPreProcessor(new ChineseTokenizer());

        //Generates a word-vector from the dataset stored in resources folder
        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(2).iterations(5).layerSize(100).seed(42)
                        .learningRate(0.1).windowSize(20).iterate(iter).tokenizerFactory(tokenizerFactory).build();
        vec.fit();
        WordVectorSerializer.writeWordVectors(vec, new File("src/test/resources/chineseNameWordVector.txt"));

        //trains a model that can find out all names from news(Suffix txt),It uses word vector generated
        // WordVectors wordVectors;

        //test model,Whether the model find out name from unknow text;

    }


}
