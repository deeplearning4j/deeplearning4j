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

package org.deeplearning4j.spark.models.sequencevectors.export.impl;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * This model exporter is suitable for debug/testing only.
 *
 * PLEASE NOTE: Never use this exporter in real environment if your model won't fit into memory of driver.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class VocabCacheExporter implements SparkModelExporter<VocabWord> {

    @Getter
    protected VocabCache<VocabWord> vocabCache;
    @Getter
    protected InMemoryLookupTable<VocabWord> lookupTable;
    @Getter
    protected Word2Vec word2Vec;

    public VocabCacheExporter() {

    }

    @Override
    public void export(JavaRDD<ExportContainer<VocabWord>> rdd) {

        // beware, generally that's VERY bad idea, but will work fine for testing purposes
        List<ExportContainer<VocabWord>> list = rdd.collect();

        if (vocabCache == null)
            vocabCache = new AbstractCache<>();

        INDArray syn0 = null;

        // just roll through list
        for (ExportContainer<VocabWord> element : list) {
            VocabWord word = element.getElement();
            INDArray weights = element.getArray();

            if (syn0 == null)
                syn0 = Nd4j.create(list.size(), weights.length());


            vocabCache.addToken(word);
            vocabCache.addWordToIndex(word.getIndex(), word.getLabel());


            syn0.getRow(word.getIndex()).assign(weights);
        }

        if (lookupTable == null)
            lookupTable = new InMemoryLookupTable.Builder<VocabWord>().cache(vocabCache).vectorLength(syn0.columns())
                            .build();

        lookupTable.setSyn0(syn0);

        // this is bad & dirty, but we don't really need anything else for testing :)
        word2Vec = WordVectorSerializer
                        .fromPair(Pair.<InMemoryLookupTable, VocabCache>makePair(lookupTable, vocabCache));
    }
}
