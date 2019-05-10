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

package org.deeplearning4j.models.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

/**
 * This set of tests will address TSNE building checks, as well as parts of UI package involved there
 *
 *
 * @author raver119@gmail.com
 */
@Ignore
public class Word2VecVisualizationTests {

    private static WordVectors vectors;

    @Before
    public synchronized void setUp() throws Exception {
        if (vectors == null) {
            vectors = WordVectorSerializer.loadFullModel("/ext/Temp/Models/raw_sentences.dat");
        }
    }

    @Test
    public void testBarnesHutTsneVisualization() throws Exception {
        BarnesHutTsne tsne = new BarnesHutTsne.Builder().setMaxIter(4).stopLyingIteration(250).learningRate(500)
                        .useAdaGrad(false).theta(0.5).setMomentum(0.5).normalize(true).build();

        //vectors.lookupTable().plotVocab(tsne);
    }
}
