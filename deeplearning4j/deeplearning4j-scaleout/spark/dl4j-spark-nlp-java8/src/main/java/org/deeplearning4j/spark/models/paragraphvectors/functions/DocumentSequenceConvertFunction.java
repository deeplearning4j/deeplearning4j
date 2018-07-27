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

package org.deeplearning4j.spark.models.paragraphvectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.spark.models.sequencevectors.functions.BaseTokenizerFunction;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class DocumentSequenceConvertFunction extends BaseTokenizerFunction
                implements Function<LabelledDocument, Sequence<VocabWord>> {

    public DocumentSequenceConvertFunction(@NonNull Broadcast<VectorsConfiguration> configurationBroadcast) {
        super(configurationBroadcast);
    }

    @Override
    public Sequence<VocabWord> call(LabelledDocument document) throws Exception {
        Sequence<VocabWord> sequence = new Sequence<>();

        // get elements
        if (document.getReferencedContent() != null && !document.getReferencedContent().isEmpty()) {
            sequence.addElements(document.getReferencedContent());
        } else {
            if (tokenizerFactory == null)
                instantiateTokenizerFactory();

            List<String> tokens = tokenizerFactory.create(document.getContent()).getTokens();

            for (String token : tokens) {
                if (token == null || token.isEmpty())
                    continue;

                VocabWord word = new VocabWord(1.0, token);
                sequence.addElement(word);
            }
        }

        // get labels
        for (String label : document.getLabels()) {
            if (label == null || label.isEmpty())
                continue;

            VocabWord labelElement = new VocabWord(1.0, label);
            labelElement.markAsLabel(true);

            sequence.addSequenceLabel(labelElement);
        }

        return sequence;
    }
}
