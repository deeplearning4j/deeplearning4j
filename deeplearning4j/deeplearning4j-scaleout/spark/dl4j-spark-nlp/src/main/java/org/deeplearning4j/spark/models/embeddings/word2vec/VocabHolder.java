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

package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * @author raver119@gmail.com
 */
public class VocabHolder implements Serializable {
    private static VocabHolder ourInstance = new VocabHolder();

    private Map<VocabWord, INDArray> indexSyn0VecMap = new ConcurrentHashMap<>();
    private Map<Integer, INDArray> pointSyn1VecMap = new ConcurrentHashMap<>();
    private HashSet<Long> workers = new LinkedHashSet<>();

    private AtomicLong seed = new AtomicLong(0);
    private AtomicInteger vectorLength = new AtomicInteger(0);

    public static VocabHolder getInstance() {
        return ourInstance;
    }

    private VocabHolder() {}

    public void setSeed(long seed, int vectorLength) {
        this.seed.set(seed);
        this.vectorLength.set(vectorLength);
    }

    public INDArray getSyn0Vector(Integer wordIndex, VocabCache<VocabWord> vocabCache) {
        if (!workers.contains(Thread.currentThread().getId()))
            workers.add(Thread.currentThread().getId());

        VocabWord word = vocabCache.elementAtIndex(wordIndex);

        if (!indexSyn0VecMap.containsKey(word)) {
            synchronized (this) {
                if (!indexSyn0VecMap.containsKey(word)) {
                    indexSyn0VecMap.put(word, getRandomSyn0Vec(vectorLength.get(), wordIndex));
                }
            }
        }

        return indexSyn0VecMap.get(word);
    }

    public INDArray getSyn1Vector(Integer point) {

        if (!pointSyn1VecMap.containsKey(point)) {
            synchronized (this) {
                if (!pointSyn1VecMap.containsKey(point)) {
                    pointSyn1VecMap.put(point, Nd4j.zeros(1, vectorLength.get()));
                }
            }
        }

        return pointSyn1VecMap.get(point);
    }

    private INDArray getRandomSyn0Vec(int vectorLength, long lseed) {
        /*
            we use wordIndex as part of seed here, to guarantee that during word syn0 initialization on dwo distinct nodes, initial weights will be the same for the same word
         */
        return Nd4j.rand(lseed * seed.get(), new int[] {1, vectorLength}).subi(0.5).divi(vectorLength);
    }

    public Iterable<Map.Entry<VocabWord, INDArray>> getSplit(VocabCache<VocabWord> vocabCache) {
        Set<Map.Entry<VocabWord, INDArray>> set = new HashSet<>();
        int cnt = 0;
        for (Map.Entry<VocabWord, INDArray> entry : indexSyn0VecMap.entrySet()) {
            set.add(entry);
            cnt++;
            if (cnt > 10)
                break;
        }

        System.out.println("Returning set: " + set.size());

        return set;
    }
}
