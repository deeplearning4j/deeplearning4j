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

package org.datavec.api.transform.ops;

import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.common.tests.BaseND4JTest;

import java.util.*;

import static org.junit.Assert.assertTrue;

/**
 * Created by huitseeker on 5/14/17.
 */
public class AggregableMultiOpTest extends BaseND4JTest {

    private List<Integer> intList = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));

    @Test
    public void testMulti() throws Exception {
        AggregatorImpls.AggregableFirst<Integer> af = new AggregatorImpls.AggregableFirst<>();
        AggregatorImpls.AggregableSum<Integer> as = new AggregatorImpls.AggregableSum<>();
        AggregableMultiOp<Integer> multi = new AggregableMultiOp<>(Arrays.asList(af, as));

        assertTrue(multi.getOperations().size() == 2);
        for (int i = 0; i < intList.size(); i++) {
            multi.accept(intList.get(i));
        }

        // mutablility
        assertTrue(as.get().toDouble() == 45D);
        assertTrue(af.get().toInt() == 1);

        List<Writable> res = multi.get();
        assertTrue(res.get(1).toDouble() == 45D);
        assertTrue(res.get(0).toInt() == 1);

        AggregatorImpls.AggregableFirst<Integer> rf = new AggregatorImpls.AggregableFirst<>();
        AggregatorImpls.AggregableSum<Integer> rs = new AggregatorImpls.AggregableSum<>();
        AggregableMultiOp<Integer> reverse = new AggregableMultiOp<>(Arrays.asList(rf, rs));

        for (int i = 0; i < intList.size(); i++) {
            reverse.accept(intList.get(intList.size() - i - 1));
        }

        List<Writable> revRes = reverse.get();
        assertTrue(revRes.get(1).toDouble() == 45D);
        assertTrue(revRes.get(0).toInt() == 9);

        multi.combine(reverse);
        List<Writable> combinedRes = multi.get();
        assertTrue(combinedRes.get(1).toDouble() == 90D);
        assertTrue(combinedRes.get(0).toInt() == 1);
    }
}
