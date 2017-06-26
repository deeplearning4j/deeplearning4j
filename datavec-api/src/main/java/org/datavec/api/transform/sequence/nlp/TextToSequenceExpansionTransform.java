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

import lombok.NonNull;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.nlp.Tokenizer;
import org.datavec.api.transform.nlp.TokenizerFactory;
import org.datavec.api.transform.nlp.VocabProvider;
import org.datavec.api.transform.nlp.Vocabulary;
import org.datavec.api.transform.nlp.impl.DefaultTokenizerFactory;
import org.datavec.api.transform.sequence.expansion.BaseSequenceExpansionTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Alex on 26/06/2017.
 */
public class TextToSequenceExpansionTransform extends BaseSequenceExpansionTransform {

    private VocabProvider vocabProvider;
    private TokenizerFactory tokenizerFactory;

    public TextToSequenceExpansionTransform(@NonNull String textColumn, @NonNull VocabProvider vocabProvider) {
        this(textColumn, textColumn, vocabProvider);
    }

    public TextToSequenceExpansionTransform(@NonNull String textColumn, @NonNull String newColumnName, @NonNull VocabProvider vocabProvider) {
        this(textColumn, newColumnName, vocabProvider, new DefaultTokenizerFactory(true));
    }


    public TextToSequenceExpansionTransform(@JsonProperty("textColumn") @NonNull String textColumn,
                                            @JsonProperty("newColumnName") @NonNull String newColumnName,
                                            @JsonProperty("vocabProvider") @NonNull VocabProvider vocabProvider,
                                            @JsonProperty("tokenizerFactory") @NonNull TokenizerFactory tokenizerFactory) {
        super(Collections.singletonList(textColumn), Collections.singletonList(newColumnName));

        this.vocabProvider = vocabProvider;
        this.tokenizerFactory = tokenizerFactory;
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
