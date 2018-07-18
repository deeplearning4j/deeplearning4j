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

package org.deeplearning4j.parallelism;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class AsyncIteratorTest extends BaseDL4JTest {

    @Test
    public void hasNext() throws Exception {
        ArrayList<Integer> integers = new ArrayList<>();
        for (int x = 0; x < 100000; x++) {
            integers.add(x);
        }

        AsyncIterator<Integer> iterator = new AsyncIterator<>(integers.iterator(), 512);
        int cnt = 0;
        Integer val = null;
        while (iterator.hasNext()) {
            val = iterator.next();
            assertEquals(cnt, val.intValue());
            cnt++;
        }

        System.out.println("Last val: " + val);

        assertEquals(integers.size(), cnt);
    }

}
