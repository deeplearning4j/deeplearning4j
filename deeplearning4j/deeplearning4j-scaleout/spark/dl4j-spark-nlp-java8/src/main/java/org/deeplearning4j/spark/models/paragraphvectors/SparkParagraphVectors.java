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

package org.deeplearning4j.spark.models.paragraphvectors;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.paragraphvectors.functions.DocumentSequenceConvertFunction;
import org.deeplearning4j.spark.models.paragraphvectors.functions.KeySequenceConvertFunction;
import org.deeplearning4j.spark.models.sequencevectors.SparkSequenceVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

/**
 * @author raver119@gmail.com
 */
public class SparkParagraphVectors extends SparkSequenceVectors<VocabWord> {

    protected SparkParagraphVectors() {
        //
    }

    @Override
    protected VocabCache<ShallowSequenceElement> getShallowVocabCache() {
        return super.getShallowVocabCache();
    }

    @Override
    protected void validateConfiguration() {
        super.validateConfiguration();

        if (configuration.getTokenizerFactory() == null)
            throw new DL4JInvalidConfigException(
                            "TokenizerFactory is undefined. Can't train ParagraphVectors without it.");
    }

    /**
     * This method builds ParagraphVectors model, expecting JavaPairRDD with key as label, and value as document-in-a-string.
     *
     * @param documentsRdd
     */
    public void fitMultipleFiles(JavaPairRDD<String, String> documentsRdd) {
        /*
            All we want here, is to transform JavaPairRDD into JavaRDD<Sequence<VocabWord>>
         */
        validateConfiguration();

        broadcastEnvironment(new JavaSparkContext(documentsRdd.context()));

        JavaRDD<Sequence<VocabWord>> sequenceRdd =
                        documentsRdd.map(new KeySequenceConvertFunction(configurationBroadcast));

        super.fitSequences(sequenceRdd);
    }

    /**
     * This method builds ParagraphVectors model, expecting JavaRDD<LabelledDocument>.
     * It can be either non-tokenized documents, or tokenized.
     *
     * @param documentsRdd
     */
    public void fitLabelledDocuments(JavaRDD<LabelledDocument> documentsRdd) {

        validateConfiguration();

        broadcastEnvironment(new JavaSparkContext(documentsRdd.context()));

        JavaRDD<Sequence<VocabWord>> sequenceRDD =
                        documentsRdd.map(new DocumentSequenceConvertFunction(configurationBroadcast));

        super.fitSequences(sequenceRDD);
    }

}
