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

package org.deeplearning4j.spark.models.sequencevectors.export;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class ExportContainerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testToString() throws Exception {
        ExportContainer<VocabWord> container =
                        new ExportContainer<>(new VocabWord(1.0, "word"), Nd4j.create(new double[] {1.01, 2.01, 3.01}));
        String exp = "word 1.01 2.01 3.01";
        String string = container.toString();

        assertEquals(exp, string);
    }

}
