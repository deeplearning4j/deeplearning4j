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

package org.datavec.api.transform.ops;

import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * Created by huitseeker on 5/14/17.
 */
public class DispatchOpTest {

    private List<Integer> intList = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
    private List<String> stringList = new ArrayList<>(Arrays.asList("arakoa", "abracadabra", "blast", "acceptance"));

    @Test
    public void testDispatchSimple() {
        AggregatorImpls.AggregableFirst<Integer> af = new AggregatorImpls.AggregableFirst<>();
        AggregatorImpls.AggregableSum<Integer> as = new AggregatorImpls.AggregableSum<>();
        AggregableMultiOp<Integer> multiaf =
                        new AggregableMultiOp<>(Collections.<IAggregableReduceOp<Integer, Writable>>singletonList(af));
        AggregableMultiOp<Integer> multias =
                        new AggregableMultiOp<>(Collections.<IAggregableReduceOp<Integer, Writable>>singletonList(as));

        DispatchOp<Integer, Writable> parallel =
                        new DispatchOp<>(Arrays.<IAggregableReduceOp<Integer, List<Writable>>>asList(multiaf, multias));

        assertTrue(multiaf.getOperations().size() == 1);
        assertTrue(multias.getOperations().size() == 1);
        assertTrue(parallel.getOperations().size() == 2);
        for (int i = 0; i < intList.size(); i++) {
            parallel.accept(Arrays.asList(intList.get(i), intList.get(i)));
        }

        List<Writable> res = parallel.get();
        assertTrue(res.get(1).toDouble() == 45D);
        assertTrue(res.get(0).toInt() == 1);

    }

    @Test
    public void testDispatchFlatMap() {
        AggregatorImpls.AggregableFirst<Integer> af = new AggregatorImpls.AggregableFirst<>();
        AggregatorImpls.AggregableSum<Integer> as = new AggregatorImpls.AggregableSum<>();
        AggregableMultiOp<Integer> multi = new AggregableMultiOp<>(Arrays.asList(af, as));

        AggregatorImpls.AggregableLast<Integer> al = new AggregatorImpls.AggregableLast<>();
        AggregatorImpls.AggregableMax<Integer> amax = new AggregatorImpls.AggregableMax<>();
        AggregableMultiOp<Integer> otherMulti = new AggregableMultiOp<>(Arrays.asList(al, amax));


        DispatchOp<Integer, Writable> parallel = new DispatchOp<>(
                        Arrays.<IAggregableReduceOp<Integer, List<Writable>>>asList(multi, otherMulti));

        assertTrue(multi.getOperations().size() == 2);
        assertTrue(otherMulti.getOperations().size() == 2);
        assertTrue(parallel.getOperations().size() == 2);
        for (int i = 0; i < intList.size(); i++) {
            parallel.accept(Arrays.asList(intList.get(i), intList.get(i)));
        }

        List<Writable> res = parallel.get();
        assertTrue(res.get(1).toDouble() == 45D);
        assertTrue(res.get(0).toInt() == 1);
        assertTrue(res.get(3).toDouble() == 9);
        assertTrue(res.get(2).toInt() == 9);

    }

}
