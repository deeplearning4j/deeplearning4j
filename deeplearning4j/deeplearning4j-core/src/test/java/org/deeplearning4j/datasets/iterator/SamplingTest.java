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

package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
public class SamplingTest extends BaseDL4JTest {

    @Test
    public void testSample() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(10, 10);
        //batch size and total
        DataSetIterator sampling = new SamplingDataSetIterator(iter.next(), 10, 10);
        assertEquals(sampling.next().numExamples(), 10);
    }

}
