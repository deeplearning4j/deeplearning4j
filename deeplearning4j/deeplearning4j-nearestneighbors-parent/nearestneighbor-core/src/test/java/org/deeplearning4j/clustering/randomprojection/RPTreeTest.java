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

package org.deeplearning4j.clustering.randomprojection;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.util.List;

import static org.junit.Assert.*;

public class RPTreeTest {


    @Test
    public void testRPTree() throws Exception {
        DataSetIterator mnist = new MnistDataSetIterator(150,150);
        RPTree rpTree = new RPTree(784,50);
        DataSet d = mnist.next();
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(d);
        normalizerStandardize.transform(d.getFeatures());
        INDArray data = d.getFeatures();
        rpTree.buildTree(data);
        assertEquals(4,rpTree.getLeaves().size());
        assertEquals(0,rpTree.getRoot().getDepth());

        List<Integer> candidates = rpTree.getCandidates(data.getRow(0));
        assertFalse(candidates.isEmpty());
        assertEquals(10,rpTree.query(data.slice(0),10).length());
        System.out.println(candidates.size());

        rpTree.addNodeAtIndex(150,data.getRow(0));

    }

    @Test
    public void testFindSelf() throws Exception {
        DataSetIterator mnist = new MnistDataSetIterator(100, 6000);
        NormalizerMinMaxScaler minMaxNormalizer = new NormalizerMinMaxScaler(0, 1);
        minMaxNormalizer.fit(mnist);
        DataSet d = mnist.next();
        minMaxNormalizer.transform(d.getFeatures());
        RPForest rpForest = new RPForest(100, 100, "euclidean");
        rpForest.fit(d.getFeatures());
        for (int i = 0; i < 10; i++) {
            INDArray indexes = rpForest.queryAll(d.getFeatures().slice(i), 10);
            assertEquals(i,indexes.getInt(0));
        }
    }

    @Test
    public void testRpTreeMaxNodes() throws Exception {
        DataSetIterator mnist = new MnistDataSetIterator(150,150);
        RPForest rpTree = new RPForest(4,4,"euclidean");
        DataSet d = mnist.next();
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(d);
        rpTree.fit(d.getFeatures());
        for(RPTree tree : rpTree.getTrees()) {
            for(RPNode node : tree.getLeaves()) {
                assertTrue(node.getIndices().size() <= rpTree.getMaxSize());
            }
        }

    }


}
