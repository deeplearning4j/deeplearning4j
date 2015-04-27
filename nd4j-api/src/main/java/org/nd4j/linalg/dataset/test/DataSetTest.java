/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.dataset.test;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertTrue;

/**
 * @author Adam Gibson
 */
public abstract class DataSetTest {

    private static Logger log = LoggerFactory.getLogger(DataSetTest.class);

    @Test
    public void testFilterAndStrip() {
        INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{0, 1, 2, 1, 2, 2, 0, 1, 2, 1}, 3);

        DataSet d = new org.nd4j.linalg.dataset.DataSet(Nd4j.ones(10, 2), labels);

        //strip the dataset down to just these labels. Rearrange them such that each label is in the specified position.
        d.filterAndStrip(new int[]{1, 2});

        for (int i = 0; i < d.numExamples(); i++) {
            int outcome = d.get(i).outcome();
            assertTrue(outcome == 0 || outcome == 1);
        }


    }

}
