/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.datasets.iterator;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

@Slf4j
public class TestEmnistDataSetIterator extends BaseDL4JTest {

    @Rule
    public Timeout timeout = Timeout.seconds(600);

    @Override
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Test
    public void testEmnistDataSetIterator() throws Exception {


        int batchSize = 128;

        EmnistDataSetIterator.Set[] sets;
        if(isIntegrationTests()){
            sets = EmnistDataSetIterator.Set.values();
        } else {
            sets = new EmnistDataSetIterator.Set[]{EmnistDataSetIterator.Set.MNIST, EmnistDataSetIterator.Set.LETTERS};
        }

        for (EmnistDataSetIterator.Set s : sets) {
            boolean isBalanced = EmnistDataSetIterator.isBalanced(s);
            int numLabels = EmnistDataSetIterator.numLabels(s);
            INDArray labelCounts = null;
            for (boolean train : new boolean[] {true, false}) {
                if (isBalanced && train) {
                    labelCounts = Nd4j.create(numLabels);
                } else {
                    labelCounts = null;
                }

                log.info("Starting test: {}, {}", s, (train ? "train" : "test"));
                EmnistDataSetIterator iter = new EmnistDataSetIterator(s, batchSize, train, 12345);

                assertTrue(iter.asyncSupported());
                assertTrue(iter.resetSupported());

                int expNumExamples;
                if (train) {
                    expNumExamples = EmnistDataSetIterator.numExamplesTrain(s);
                } else {
                    expNumExamples = EmnistDataSetIterator.numExamplesTest(s);
                }



                assertEquals(numLabels, iter.getLabels().size());
                assertEquals(numLabels, iter.getLabelsArrays().length);

                char[] labelArr = iter.getLabelsArrays();
                for (char c : labelArr) {
                    boolean isExpected = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                    assertTrue(isExpected);
                }

                int totalCount = 0;
                while (iter.hasNext()) {
                    DataSet ds = iter.next();
                    assertNotNull(ds.getFeatures());
                    assertNotNull(ds.getLabels());
                    assertEquals(ds.getFeatures().size(0), ds.getLabels().size(0));

                    totalCount += ds.getFeatures().size(0);

                    assertEquals(784, ds.getFeatures().size(1));
                    assertEquals(numLabels, ds.getLabels().size(1));

                    if (isBalanced && train) {
                        labelCounts.addi(ds.getLabels().sum(0));
                    }
                }

                assertEquals(expNumExamples, totalCount);

                if (isBalanced && train) {
                    int min = labelCounts.minNumber().intValue();
                    int max = labelCounts.maxNumber().intValue();
                    int exp = expNumExamples / numLabels;

                    assertTrue(min > 0);
                    assertEquals(exp, min);
                    assertEquals(exp, max);
                }
            }
        }
    }
}
