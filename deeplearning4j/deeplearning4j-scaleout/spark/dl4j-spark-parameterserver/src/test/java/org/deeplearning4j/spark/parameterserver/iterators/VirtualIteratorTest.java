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

package org.deeplearning4j.spark.parameterserver.iterators;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class VirtualIteratorTest {
    @Before
    public void setUp() throws Exception {
        //
    }

    @Test
    public void testIteration1() throws Exception {
        List<Integer> integers = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            integers.add(i);
        }

        VirtualIterator<Integer> virt = new VirtualIterator<>(integers.iterator());

        int cnt = 0;
        while (virt.hasNext()) {
            Integer n = virt.next();
            assertEquals(cnt, n.intValue());
            cnt++;
        }


        assertEquals(100, cnt);
    }
}
