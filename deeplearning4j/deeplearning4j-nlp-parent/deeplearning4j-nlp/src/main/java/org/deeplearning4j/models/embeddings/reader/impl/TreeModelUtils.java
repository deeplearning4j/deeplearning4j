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

package org.deeplearning4j.models.embeddings.reader.impl;

import lombok.NonNull;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.util.SetUtils;

import java.util.*;

/**
 * This is VPTree-based implementation for wordsNearest method, suited for multiple consequent calls.
 * Please note: VPTree will take some memory, dependant on your model size.
 *
 * @author raver119@gmail.com
 */
public class TreeModelUtils<T extends SequenceElement> extends BasicModelUtils<T> {
    protected VPTree vpTree;

    @Override
    public void init(@NonNull WeightLookupTable<T> lookupTable) {
        super.init(lookupTable);
        vpTree = null;
    }

    protected synchronized void checkTree() {
        // build new tree if it wasn't created before
        if (vpTree == null) {
            List<DataPoint> points = new ArrayList<>();
            for (String word : vocabCache.words()) {
                points.add(new DataPoint(vocabCache.indexOf(word), lookupTable.vector(word)));
            }
            vpTree = new VPTree(points);
        }
    }


    /**
     * This method returns nearest words for target word, based on tree structure.
     * This method is recommended to use if you're going to call for nearest words multiple times.
     * VPTree will be built upon firt call to this method
     *
     * @param label label of element we're looking nearest words to
     * @param n number of nearest elements to return
     * @return
     */
    @Override
    public Collection<String> wordsNearest(String label, int n) {
        if (!vocabCache.hasToken(label))
            return new ArrayList<>();

        Collection<String> collection = wordsNearest(Arrays.asList(label), new ArrayList<String>(), n + 1);
        if (collection.contains(label))
            collection.remove(label);

        return collection;
    }

    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {

        // Check every word is in the model
        for (String p : SetUtils.union(new HashSet<>(positive), new HashSet<>(negative))) {
            if (!vocabCache.containsWord(p)) {
                return new ArrayList<>();
            }
        }

        INDArray words = Nd4j.create(positive.size() + negative.size(), lookupTable.layerSize());
        int row = 0;
        for (String s : positive) {
            words.putRow(row++, lookupTable.vector(s));
        }

        for (String s : negative) {
            words.putRow(row++, lookupTable.vector(s).mul(-1));
        }

        INDArray mean = words.isMatrix() ? words.mean(0) : words;

        return wordsNearest(mean, top);
    }

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        checkTree();

        List<DataPoint> add = new ArrayList<>();
        List<Double> distances = new ArrayList<>();

        // we need n+1 to address original datapoint removal
        vpTree.search(words, top, add, distances);

        Collection<String> ret = new ArrayList<>();
        for (DataPoint e : add) {
            String word = vocabCache.wordAtIndex(e.getIndex());
            ret.add(word);
        }

        return super.wordsNearest(words, top);
    }
}
