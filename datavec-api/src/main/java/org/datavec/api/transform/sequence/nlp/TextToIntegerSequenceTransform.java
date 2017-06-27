/*
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.datavec.api.transform.sequence.nlp;

import lombok.Data;
import lombok.NonNull;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.nlp.Tokenizer;
import org.datavec.api.transform.nlp.TokenizerFactory;
import org.datavec.api.transform.nlp.VocabProvider;
import org.datavec.api.transform.nlp.Vocabulary;
import org.datavec.api.transform.nlp.impl.DefaultTokenizerFactory;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.expansion.BaseSequenceExpansionTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The text to sequence expansion transform takes a sequence and expands 1 of the text columns into a number of steps,
 * using a vocabulary to assign an integer index to each word. All other column values are simply duplicated.<br>
 *
 * For example, if the input sequence as a single step with values [1, 2, "some text here"]<br>
 * And vocabulary is [..., 200="some", ..., 352="text", ..., 971="here"]<br>
 * Then the output will be:<br>
 * [1, 2, 200]<br>
 * [1, 2, 352]<br>
 * [1, 2, 971]<br>
 * <br>
 * Note that when the input sequence has more than one step, each step is expanded separately, in exactly the same
 * manner, and are concatenated.<br>
 * Note also that words not appearing in the vocabulary are excluded from the expanded representation.
 *
 * @author Alex Black
 */
@Data
public class TextToIntegerSequenceTransform extends BaseSequenceExpansionTransform {

    private VocabProvider vocabProvider;
    private TokenizerFactory tokenizerFactory;

    /**
     * @param textColumn    Column containing the text to expand
     * @param vocabProvider Vocabulary provider (for String to
     */
    public TextToIntegerSequenceTransform(@NonNull String textColumn, @NonNull VocabProvider vocabProvider) {
        this(textColumn, textColumn, vocabProvider);
    }

    /**
     * @param textColumn    Column containing the text to expand
     * @param newColumnName Name of the text column, after processing (becomes an integer column)
     * @param vocabProvider Vocabulary provider
     */
    public TextToIntegerSequenceTransform(@NonNull String textColumn, @NonNull String newColumnName, @NonNull VocabProvider vocabProvider) {
        this(textColumn, newColumnName, vocabProvider, new DefaultTokenizerFactory(true));
    }

    /**
     * @param textColumn       Column containing the text to expand
     * @param newColumnName    Name of the text column, after processing (becomes an integer column)
     * @param vocabProvider    Vocabulary provider
     * @param tokenizerFactory Tokenizer factory, for controlling how tokenization is done
     */
    public TextToIntegerSequenceTransform(@NonNull String textColumn,
                                          @NonNull String newColumnName,
                                          @NonNull VocabProvider vocabProvider,
                                          @NonNull TokenizerFactory tokenizerFactory) {
        this(Collections.singletonList(textColumn), Collections.singletonList(newColumnName), vocabProvider, tokenizerFactory);
    }

    //Private constructor for JSON ser/de
    private TextToIntegerSequenceTransform(@JsonProperty("requiredColumns") @NonNull List<String> requiredColumns,
                                          @JsonProperty("expandedColumnNames") @NonNull List<String> expandedColumnNames,
                                          @JsonProperty("vocabProvider") @NonNull VocabProvider vocabProvider,
                                          @JsonProperty("tokenizerFactory") @NonNull TokenizerFactory tokenizerFactory) {
        super(requiredColumns, expandedColumnNames);

        this.vocabProvider = vocabProvider;
        this.tokenizerFactory = tokenizerFactory;
    }

    @Override
    public void setInputSchema(Schema schema){
        if(!schema.hasColumn(requiredColumns.get(0))){
            throw new IllegalStateException("String/Text column \"" + requiredColumns.get(0) + "\" not found in schema");
        }
        if(schema.getType(requiredColumns.get(0)) != ColumnType.String){
            throw new IllegalStateException("Column \"" + requiredColumns.get(0) + "\" is not a text column. Type: " +
                    schema.getType(requiredColumns.get(0)));
        }
        super.setInputSchema(schema);
    }

    @Override
    protected List<ColumnMetaData> expandedColumnMetaDatas(List<ColumnMetaData> origColumnMeta, List<String> expandedColumnNames) {
        //From String column to integer column

        return Collections.<ColumnMetaData>singletonList(
                new IntegerMetaData(expandedColumnNames.get(0), 0, vocabProvider.getVocab().size()));
    }

    @Override
    protected List<List<Writable>> expandTimeStep(List<Writable> currentStepValues) {
        //Expect a single text writable only
        String text = currentStepValues.get(0).toString();
        Tokenizer t = tokenizerFactory.create(text);
        List<String> tokens = t.getTokens();

        Vocabulary v = vocabProvider.getVocab();

        List<List<Writable>> out = new ArrayList<>(tokens.size());
        for(String s : tokens){
            if(v.containsWord(s)){
                out.add(Collections.<Writable>singletonList(new IntWritable(v.indexOf(s))));
            }
        }

        return out;
    }
}
