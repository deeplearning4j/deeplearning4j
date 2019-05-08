/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.nlp.tokenization.tokenizer.TokenPreProcess;
import org.datavec.nlp.tokenization.tokenizer.Tokenizer;
import org.datavec.nlp.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.datavec.nlp.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This transform takes in a list of words
 * and outputs a single vector where that vector is of size
 * number of words in the vocab.
 *
 * For more information on vocab, see {@link BagOfWordsTransform}
 *
 * For definition of a vocab, it is generated using a {@link TokenizerFactory}
 * This transform will use {@link DefaultTokenizerFactory}
 * for the tokenizer factory if one is not specified.
 * Otherwise, one can specify a custom {@link TokenizerFactory}
 * with a default constructor.
 *
 * The other components that need to be specified are:
 * a word index map representing what words go in what columns
 * an inverse document frequency map representing the weighting of inverse document frequencies
 * for each word (this is for tfidf calculation)
 *
 * This is typically used for non english languages.
 *
 *
 * @author Adam Gibson
 */
@Data
@EqualsAndHashCode(callSuper = true, exclude = {"tokenizerFactory"})
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TokenizerBagOfWordsTermSequenceIndexTransform extends BaseColumnTransform {

    private String newColumName;
    private  Map<String,Integer> wordIndexMap;
    private Map<String,Double> weightMap;
    private boolean exceptionOnUnknown;
    private String tokenizerFactoryClass;
    private String preprocessorClass;
    private TokenizerFactory tokenizerFactory;

    @JsonCreator
    public TokenizerBagOfWordsTermSequenceIndexTransform(@JsonProperty("columnName") String columnName,
                                                         @JsonProperty("newColumnName") String newColumnName,
                                                         @JsonProperty("wordIndexMap") Map<String,Integer> wordIndexMap,
                                                         @JsonProperty("idfMap") Map<String,Double> idfMap,
                                                         @JsonProperty("exceptionOnUnknown") boolean exceptionOnUnknown,
                                                         @JsonProperty("tokenizerFactoryClass") String tokenizerFactoryClass,
                                                         @JsonProperty("preprocessorClass") String preprocessorClass) {
        super(columnName);
        this.newColumName = newColumnName;
        this.wordIndexMap = wordIndexMap;
        this.exceptionOnUnknown = exceptionOnUnknown;
        this.weightMap = idfMap;
        this.tokenizerFactoryClass = tokenizerFactoryClass;
        this.preprocessorClass = preprocessorClass;
        if(this.tokenizerFactoryClass == null) {
            this.tokenizerFactoryClass = DefaultTokenizerFactory.class.getName();
        }
        try {
            tokenizerFactory = (TokenizerFactory) Class.forName(this.tokenizerFactoryClass).newInstance();
        } catch (Exception e) {
            throw new IllegalStateException("Unable to instantiate tokenizer factory with empty constructor. Does the tokenizer factory class contain a default empty constructor?");
        }

        if(preprocessorClass != null){
            try {
                TokenPreProcess tpp = (TokenPreProcess) Class.forName(this.preprocessorClass).newInstance();
                tokenizerFactory.setTokenPreProcessor(tpp);
            } catch (Exception e){
                throw new IllegalStateException("Unable to instantiate preprocessor factory with empty constructor. Does the tokenizer factory class contain a default empty constructor?");
            }
        }

    }



    @Override
    public List<Writable> map(List<Writable> writables) {
        Text text = (Text) writables.get(inputSchema.getIndexOfColumn(columnName));
        List<Writable> ret = new ArrayList<>(writables);
        ret.set(inputSchema.getIndexOfColumn(columnName),new NDArrayWritable(convert(text.toString())));
        return ret;
    }

    @Override
    public Object map(Object input) {
        return convert(input.toString());
    }

    @Override
    public Object mapSequence(Object sequence) {
        return convert(sequence.toString());
    }

    @Override
    public Schema transform(Schema inputSchema) {
        Schema.Builder newSchema = new Schema.Builder();
        for(int i = 0; i < inputSchema.numColumns(); i++) {
            if(inputSchema.getName(i).equals(this.columnName)) {
                newSchema.addColumnNDArray(newColumName,new long[]{1,wordIndexMap.size()});
            }
            else {
                newSchema.addColumn(inputSchema.getMetaData(i));
            }
        }

        return newSchema.build();
    }


    /**
     * Convert the given text
     * in to an {@link INDArray}
     * using the {@link TokenizerFactory}
     * specified in the constructor.
     * @param text the text to transform
     * @return the created {@link INDArray}
     * based on the {@link #wordIndexMap} for the column indices
     * of the word.
     */
    public INDArray convert(String text) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray create = Nd4j.create(1,wordIndexMap.size());
        Counter<String> tokenizedCounter = new Counter<>();

        for(int i = 0; i < tokens.size(); i++) {
            tokenizedCounter.incrementCount(tokens.get(i),1.0);
        }

        for(int i = 0; i < tokens.size(); i++) {
            if(wordIndexMap.containsKey(tokens.get(i))) {
                int idx = wordIndexMap.get(tokens.get(i));
                int count = (int) tokenizedCounter.getCount(tokens.get(i));
                double weight = tfidfWord(tokens.get(i),count,tokens.size());
                create.putScalar(idx,weight);
            }
        }

        return create;
    }


    /**
     * Calculate the tifdf for a word
     * given the word, word count, and document length
     * @param word the word to calculate
     * @param wordCount the word frequency
     * @param documentLength the number of words in the document
     * @return the tfidf weight for a given word
     */
    public double tfidfWord(String word, long wordCount, long documentLength) {
        double tf = tfForWord(wordCount, documentLength);
        double idf = idfForWord(word);
        return MathUtils.tfidf(tf, idf);
    }

    /**
     * Calculate the weight term frequency for a given
     * word normalized by the dcoument length
     * @param wordCount the word frequency
     * @param documentLength the number of words in the edocument
     * @return
     */
    private double tfForWord(long wordCount, long documentLength) {
        return wordCount;
    }

    private double idfForWord(String word) {
        if(weightMap.containsKey(word))
            return weightMap.get(word);
        return 0;
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new NDArrayMetaData(outputColumnName(),new long[]{1,wordIndexMap.size()});
    }

    @Override
    public String outputColumnName() {
        return newColumName;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[]{newColumName};
    }

    @Override
    public String[] columnNames() {
        return new String[]{columnName()};
    }

    @Override
    public String columnName() {
        return columnName;
    }

    @Override
    public Writable map(Writable columnWritable) {
        return new NDArrayWritable(convert(columnWritable.toString()));
    }
}
