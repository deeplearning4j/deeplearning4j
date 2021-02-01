/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import lombok.val;
import lombok.var;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.tools.SimpleVariableGenerator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@Slf4j
public class DummyBlockDataSetIteratorTests extends BaseDL4JTest {

    @Test
    public void testBlock_1() throws Exception {
        val simpleIterator = new SimpleVariableGenerator(123, 8, 3, 3, 3);

        val iterator = new DummyBlockDataSetIterator(simpleIterator);

        assertTrue(iterator.hasAnything());
        val list = new ArrayList<DataSet>(8);

        var datasets = iterator.next(3);
        assertNotNull(datasets);
        assertEquals(3, datasets.length);
        list.addAll(Arrays.asList(datasets));



        datasets = iterator.next(3);
        assertNotNull(datasets);
        assertEquals(3, datasets.length);
        list.addAll(Arrays.asList(datasets));

        datasets = iterator.next(3);
        assertNotNull(datasets);
        assertEquals(2, datasets.length);
        list.addAll(Arrays.asList(datasets));

        for (int e = 0; e < list.size(); e++) {
            val dataset = list.get(e);

            assertEquals(e, (int) dataset.getFeatures().getDouble(0));
            assertEquals(e + 0.5, dataset.getLabels().getDouble(0), 1e-3);
        }
    }
}
