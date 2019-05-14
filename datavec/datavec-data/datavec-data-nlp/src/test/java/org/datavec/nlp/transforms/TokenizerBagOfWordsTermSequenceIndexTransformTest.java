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
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.datavec.nlp.metadata.VocabCache;
import org.datavec.nlp.tokenization.tokenizer.preprocessor.LowerCasePreProcessor;
import org.datavec.nlp.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.datavec.nlp.vectorizer.TfidfVectorizer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Triple;

import java.util.*;

import static org.datavec.nlp.vectorizer.TextVectorizer.*;
import static org.junit.Assert.assertArrayEquals;
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

        /*
         ## Reproduce with:
         from sklearn.feature_extraction.text import TfidfVectorizer
         corpus = ["This is very strange", "This is very nice"]

         ## SMOOTH = FALSE case:
         vectorizer = TfidfVectorizer(min_df=0, norm=None, smooth_idf=False)
         X = vectorizer.fit_transform(corpus)
         idf = vectorizer.idf_
         print(dict(zip(vectorizer.get_feature_names(), idf)))

         newText = ["This is very strange", "This is very nice"]
         out = vectorizer.transform(newText)
         print(out)

         {'is': 1.0, 'nice': 1.6931471805599454, 'strange': 1.6931471805599454, 'this': 1.0, 'very': 1.0}
         (0, 4)	1.0
         (0, 3)	1.0
         (0, 2)	1.6931471805599454
         (0, 0)	1.0
         (1, 4)	1.0
         (1, 3)	1.0
         (1, 1)	1.6931471805599454
         (1, 0)	1.0

         ## SMOOTH + TRUE case:
         {'is': 1.0, 'nice': 1.4054651081081644, 'strange': 1.4054651081081644, 'this': 1.0, 'very': 1.0}
          (0, 4)	1.0
          (0, 3)	1.0
          (0, 2)	1.4054651081081644
          (0, 0)	1.0
          (1, 4)	1.0
          (1, 3)	1.0
          (1, 1)	1.4054651081081644
          (1, 0)	1.0
         */

        List<List<List<Writable>>> input = new ArrayList<>();
        input.add(Arrays.asList(Arrays.<Writable>asList(new Text(corpus[0])),Arrays.<Writable>asList(new Text(corpus[1]))));

        // First: Check TfidfVectorizer vs. scikit:

        Map<String,Double> idfMapNoSmooth = new HashMap<>();
        idfMapNoSmooth.put("is",1.0);
        idfMapNoSmooth.put("nice",1.6931471805599454);
        idfMapNoSmooth.put("strange",1.6931471805599454);
        idfMapNoSmooth.put("this",1.0);
        idfMapNoSmooth.put("very",1.0);

        Map<String,Double> idfMapSmooth = new HashMap<>();
        idfMapSmooth.put("is",1.0);
        idfMapSmooth.put("nice",1.4054651081081644);
        idfMapSmooth.put("strange",1.4054651081081644);
        idfMapSmooth.put("this",1.0);
        idfMapSmooth.put("very",1.0);



        TfidfVectorizer tfidfVectorizer = new TfidfVectorizer();
        Configuration configuration = new Configuration();
        configuration.set(TOKENIZER, DefaultTokenizerFactory.class.getName());
        configuration.set(MIN_WORD_FREQUENCY,"1");
        configuration.set(STOP_WORDS,"");
        configuration.set(TfidfVectorizer.SMOOTH_IDF, "false");

        tfidfVectorizer.initialize(configuration);

        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(input.get(0));
        INDArray array = tfidfVectorizer.fitTransform(collectionRecordReader);

        INDArray expNoSmooth = Nd4j.create(DataType.FLOAT, 2, 5);
        VocabCache vc = tfidfVectorizer.getCache();
        expNoSmooth.putScalar(0, vc.wordIndex("very"), 1.0);
        expNoSmooth.putScalar(0, vc.wordIndex("this"), 1.0);
        expNoSmooth.putScalar(0, vc.wordIndex("strange"), 1.6931471805599454);
        expNoSmooth.putScalar(0, vc.wordIndex("is"), 1.0);

        expNoSmooth.putScalar(1, vc.wordIndex("very"), 1.0);
        expNoSmooth.putScalar(1, vc.wordIndex("this"), 1.0);
        expNoSmooth.putScalar(1, vc.wordIndex("nice"), 1.6931471805599454);
        expNoSmooth.putScalar(1, vc.wordIndex("is"), 1.0);

        assertEquals(expNoSmooth, array);


        //------------------------------------------------------------
        //Smooth version:
        tfidfVectorizer = new TfidfVectorizer();
        configuration = new Configuration();
        configuration.set(TOKENIZER, DefaultTokenizerFactory.class.getName());
        configuration.set(MIN_WORD_FREQUENCY,"1");
        configuration.set(STOP_WORDS,"");
        configuration.set(TfidfVectorizer.SMOOTH_IDF, "true");

        tfidfVectorizer.initialize(configuration);

        collectionRecordReader.reset();
        array = tfidfVectorizer.fitTransform(collectionRecordReader);

        INDArray expSmooth = Nd4j.create(DataType.FLOAT, 2, 5);
        expSmooth.putScalar(0, vc.wordIndex("very"), 1.0);
        expSmooth.putScalar(0, vc.wordIndex("this"), 1.0);
        expSmooth.putScalar(0, vc.wordIndex("strange"), 1.4054651081081644);
        expSmooth.putScalar(0, vc.wordIndex("is"), 1.0);

        expSmooth.putScalar(1, vc.wordIndex("very"), 1.0);
        expSmooth.putScalar(1, vc.wordIndex("this"), 1.0);
        expSmooth.putScalar(1, vc.wordIndex("nice"), 1.4054651081081644);
        expSmooth.putScalar(1, vc.wordIndex("is"), 1.0);

        assertEquals(expSmooth, array);


        //////////////////////////////////////////////////////////

        //Second: Check transform vs scikit/TfidfVectorizer

        List<String> vocab = new ArrayList<>(5);    //Arrays.asList("is","nice","strange","this","very");
        for( int i=0; i<5; i++ ){
            vocab.add(vc.wordAt(i));
        }

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
                idfMapNoSmooth,
                false,
                null, null);

        SequenceSchema.Builder sequenceSchemaBuilder = new SequenceSchema.Builder();
        sequenceSchemaBuilder.addColumnString("input");
        SequenceSchema schema = sequenceSchemaBuilder.build();
        assertEquals("input",schema.getName(0));

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .transform(tokenizerBagOfWordsTermSequenceIndexTransform)
                .build();

        List<List<List<Writable>>> execute = LocalTransformExecutor.executeSequenceToSequence(input, transformProcess);



        //System.out.println(execute);
        INDArray arr0 = ((NDArrayWritable)execute.get(0).get(0).get(0)).get();
        INDArray arr1 = ((NDArrayWritable)execute.get(0).get(1).get(0)).get();

        assertEquals(expNoSmooth.getRow(0, true), arr0);
        assertEquals(expNoSmooth.getRow(1, true), arr1);


        //--------------------------------
        //Check smooth:

        tokenizerBagOfWordsTermSequenceIndexTransform = new TokenizerBagOfWordsTermSequenceIndexTransform(
                inputColumnName,
                outputColumnName,
                wordIndexMap,
                idfMapSmooth,
                false,
                null, null);

        schema = (SequenceSchema) new SequenceSchema.Builder().addColumnString("input").build();

        transformProcess = new TransformProcess.Builder(schema)
                .transform(tokenizerBagOfWordsTermSequenceIndexTransform)
                .build();

        execute = LocalTransformExecutor.executeSequenceToSequence(input, transformProcess);

        arr0 = ((NDArrayWritable)execute.get(0).get(0).get(0)).get();
        arr1 = ((NDArrayWritable)execute.get(0).get(1).get(0)).get();

        assertEquals(expSmooth.getRow(0, true), arr0);
        assertEquals(expSmooth.getRow(1, true), arr1);



        //Test JSON serialization:

        String json = transformProcess.toJson();
        TransformProcess fromJson = TransformProcess.fromJson(json);
        assertEquals(transformProcess, fromJson);
        List<List<List<Writable>>> execute2 = LocalTransformExecutor.executeSequenceToSequence(input, fromJson);

        INDArray arr0a = ((NDArrayWritable)execute2.get(0).get(0).get(0)).get();
        INDArray arr1a = ((NDArrayWritable)execute2.get(0).get(1).get(0)).get();

        assertEquals(expSmooth.getRow(0, true), arr0a);
        assertEquals(expSmooth.getRow(1, true), arr1a);
    }

    @Test
    public void additionalTest(){
        /*
        ## To reproduce:
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [
           'This is the first document',
           'This document is the second document',
           'And this is the third one',
           'Is this the first document',
        ]
        vectorizer = TfidfVectorizer(min_df=0, norm=None, smooth_idf=False)
        X = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())

        out = vectorizer.transform(corpus)
        print(out)

        ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
          (0, 8)	1.0
          (0, 6)	1.0
          (0, 3)	1.0
          (0, 2)	1.6931471805599454
          (0, 1)	1.2876820724517808
          (1, 8)	1.0
          (1, 6)	1.0
          (1, 5)	2.386294361119891
          (1, 3)	1.0
          (1, 1)	2.5753641449035616
          (2, 8)	1.0
          (2, 7)	2.386294361119891
          (2, 6)	1.0
          (2, 4)	2.386294361119891
          (2, 3)	1.0
          (2, 0)	2.386294361119891
          (3, 8)	1.0
          (3, 6)	1.0
          (3, 3)	1.0
          (3, 2)	1.6931471805599454
          (3, 1)	1.2876820724517808
          {'and': 2.386294361119891, 'document': 1.2876820724517808, 'first': 1.6931471805599454, 'is': 1.0, 'one': 2.386294361119891, 'second': 2.386294361119891, 'the': 1.0, 'third': 2.386294361119891, 'this': 1.0}
         */

        String[] corpus = {
                "This is the first document",
                "This document is the second document",
                "And this is the third one",
                "Is this the first document"};

        TfidfVectorizer tfidfVectorizer = new TfidfVectorizer();
        Configuration configuration = new Configuration();
        configuration.set(TOKENIZER, DefaultTokenizerFactory.class.getName());
        configuration.set(MIN_WORD_FREQUENCY,"1");
        configuration.set(STOP_WORDS,"");
        configuration.set(TfidfVectorizer.SMOOTH_IDF, "false");
        configuration.set(PREPROCESSOR, LowerCasePreProcessor.class.getName());

        tfidfVectorizer.initialize(configuration);

        List<List<List<Writable>>> input = new ArrayList<>();
        //input.add(Arrays.asList(Arrays.<Writable>asList(new Text(corpus[0])),Arrays.<Writable>asList(new Text(corpus[1]))));
        List<List<Writable>> seq = new ArrayList<>();
        for(String s : corpus){
            seq.add(Collections.<Writable>singletonList(new Text(s)));
        }
        input.add(seq);

        CollectionRecordReader crr = new CollectionRecordReader(seq);
        INDArray arr = tfidfVectorizer.fitTransform(crr);

        //System.out.println(arr);
        assertArrayEquals(new long[]{4, 9}, arr.shape());

        List<String> pyVocab = Arrays.asList("and", "document", "first", "is", "one", "second", "the", "third", "this");
        List<Triple<Integer,Integer,Double>> l = new ArrayList<>();
        l.add(new Triple<>(0, 8, 1.0));
        l.add(new Triple<>(0, 6, 1.0));
        l.add(new Triple<>(0, 3, 1.0));
        l.add(new Triple<>(0, 2, 1.6931471805599454));
        l.add(new Triple<>(0, 1, 1.2876820724517808));
        l.add(new Triple<>(1, 8, 1.0));
        l.add(new Triple<>(1, 6, 1.0));
        l.add(new Triple<>(1, 5, 2.386294361119891));
        l.add(new Triple<>(1, 3, 1.0));
        l.add(new Triple<>(1, 1, 2.5753641449035616));
        l.add(new Triple<>(2, 8, 1.0));
        l.add(new Triple<>(2, 7, 2.386294361119891));
        l.add(new Triple<>(2, 6, 1.0));
        l.add(new Triple<>(2, 4, 2.386294361119891));
        l.add(new Triple<>(2, 3, 1.0));
        l.add(new Triple<>(2, 0, 2.386294361119891));
        l.add(new Triple<>(3, 8, 1.0));
        l.add(new Triple<>(3, 6, 1.0));
        l.add(new Triple<>(3, 3, 1.0));
        l.add(new Triple<>(3, 2, 1.6931471805599454));
        l.add(new Triple<>(3, 1, 1.2876820724517808));

        INDArray exp = Nd4j.create(DataType.FLOAT, 4, 9);
        for(Triple<Integer,Integer,Double> t : l){
            //Work out work index, accounting for different vocab/word orders:
            int wIdx = tfidfVectorizer.getCache().wordIndex(pyVocab.get(t.getSecond()));
            exp.putScalar(t.getFirst(), wIdx, t.getThird());
        }

        assertEquals(exp, arr);


        Map<String,Double> idfWeights = new HashMap<>();
        idfWeights.put("and", 2.386294361119891);
        idfWeights.put("document", 1.2876820724517808);
        idfWeights.put("first", 1.6931471805599454);
        idfWeights.put("is", 1.0);
        idfWeights.put("one", 2.386294361119891);
        idfWeights.put("second", 2.386294361119891);
        idfWeights.put("the", 1.0);
        idfWeights.put("third", 2.386294361119891);
        idfWeights.put("this", 1.0);


        List<String> vocab = new ArrayList<>(9);    //Arrays.asList("is","nice","strange","this","very");
        for( int i=0; i<9; i++ ){
            vocab.add(tfidfVectorizer.getCache().wordAt(i));
        }

        String inputColumnName = "input";
        String outputColumnName = "output";
        Map<String,Integer> wordIndexMap = new HashMap<>();
        for(int i = 0; i < vocab.size(); i++) {
            wordIndexMap.put(vocab.get(i),i);
        }

        TokenizerBagOfWordsTermSequenceIndexTransform transform = new TokenizerBagOfWordsTermSequenceIndexTransform(
                inputColumnName,
                outputColumnName,
                wordIndexMap,
                idfWeights,
                false,
                null, LowerCasePreProcessor.class.getName());

        SequenceSchema.Builder sequenceSchemaBuilder = new SequenceSchema.Builder();
        sequenceSchemaBuilder.addColumnString("input");
        SequenceSchema schema = sequenceSchemaBuilder.build();
        assertEquals("input",schema.getName(0));

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .transform(transform)
                .build();

        List<List<List<Writable>>> execute = LocalTransformExecutor.executeSequenceToSequence(input, transformProcess);

        INDArray arr0 = ((NDArrayWritable)execute.get(0).get(0).get(0)).get();
        INDArray arr1 = ((NDArrayWritable)execute.get(0).get(1).get(0)).get();

        assertEquals(exp.getRow(0, true), arr0);
        assertEquals(exp.getRow(1, true), arr1);
    }

}
