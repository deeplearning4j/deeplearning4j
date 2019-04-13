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

package org.datavec.nlp.transforms;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.datavec.nlp.tokenization.tokenizer.DefaultTokenizer;
import org.datavec.nlp.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.datavec.nlp.vectorizer.TfidfVectorizer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.datavec.nlp.vectorizer.TextVectorizer.MIN_WORD_FREQUENCY;
import static org.datavec.nlp.vectorizer.TextVectorizer.STOP_WORDS;
import static org.datavec.nlp.vectorizer.TextVectorizer.TOKENIZER;
import static org.junit.Assert.assertEquals;

public class TokenizerBagOfWordsTermSequenceIndexTransformTest {

    @Test
    public void testSequenceExecution() {
        //credit: https://stackoverflow.com/questions/23792781/tf-idf-feature-weights-using-sklearn-feature-extraction-text-tfidfvectorizer
        String[] corpus = {
                "This is very strange".toLowerCase(),
                "This is very nice".toLowerCase()
        };
        //{'is': 1.0, 'nice': 1.4054651081081644, 'strange': 1.4054651081081644, 'this': 1.0, 'very': 1.0}

        /**
         * Reproduce with:
         * from sklearn.feature_extraction.text import TfidfVectorizer
         corpus = ["This is very strange",
         "This is very nice"]
         vectorizer = TfidfVectorizer(min_df=1)
         X = vectorizer.fit_transform(corpus)
         idf = vectorizer.idf_
         print(dict(zip(vectorizer.get_feature_names(), i
         */


        Map<String,Double> idfMap = new HashMap<>();
        idfMap.put("is",1.0);
        idfMap.put("nice",1.4054651081081644);
        idfMap.put("strange",1.4054651081081644);
        idfMap.put("this",1.0);
        idfMap.put("very",1.0);


        List<String> vocab = Arrays.asList("is","nice","strange","this","very");
        String inputColumnName = "input";
        String outputColumnName = "output";
        Map<String,Integer> wordIndexMap = new HashMap<>();
        for(int i = 0; i < vocab.size(); i++) {
            wordIndexMap.put(vocab.get(i),i);
        }


        TokenizerBagOfWordsTermSequenceIndexTransform tokenizerBagOfWordsTermSequenceIndexTransform = new TokenizerBagOfWordsTermSequenceIndexTransform(
                inputColumnName,
                outputColumnName,
                wordIndexMap,
                idfMap,
                false,
                null);

        SequenceSchema.Builder sequenceSchemaBuilder = new SequenceSchema.Builder();
        sequenceSchemaBuilder.addColumnString("input");

        SequenceSchema schema = sequenceSchemaBuilder.build();

        assertEquals("input",schema.getName(0));



        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .transform(tokenizerBagOfWordsTermSequenceIndexTransform)
                .build();



        INDArray x = Nd4j.create(new float[][]{
                {0.44832087f, 0.f, 0.63009934f, 0.44832087f, 0.44832087f},
                {0.44832087f,0.63009934f, 0.f,0.44832087f,0.44832087f}
        });


        List<List<List<Writable>>> input = new ArrayList<>();
        input.add(Arrays.asList(Arrays.<Writable>asList(new Text(corpus[0])),Arrays.<Writable>asList(new Text(corpus[1]))));
        List<List<List<Writable>>> execute = LocalTransformExecutor.executeSequenceToSequence(input, transformProcess);

        TfidfVectorizer tfidfVectorizer = new TfidfVectorizer();
        Configuration configuration = new Configuration();
        configuration.set(TOKENIZER, DefaultTokenizerFactory.class.getName());
        configuration.set(MIN_WORD_FREQUENCY,"1");
        configuration.set(STOP_WORDS,"");

        tfidfVectorizer.initialize(configuration);

        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(input.get(0));
        INDArray array = tfidfVectorizer.fitTransform(collectionRecordReader);
        System.out.println(array);

        System.out.println(execute);

    }

}
