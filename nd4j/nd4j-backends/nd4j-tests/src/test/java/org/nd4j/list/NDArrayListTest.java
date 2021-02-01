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

package org.nd4j.list;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class NDArrayListTest extends BaseNd4jTest {

    public NDArrayListTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testList() {
        NDArrayList ndArrayList = new NDArrayList();
        List<Double> arrayAssertion = new ArrayList<>();
        for(int i = 0; i < 11; i++) {
            ndArrayList.add((double) i);
            arrayAssertion.add((double) i);
        }

        assertEquals(arrayAssertion.size(),arrayAssertion.size());
        assertEquals(arrayAssertion,ndArrayList);


        arrayAssertion.remove(1);
        ndArrayList.remove(1);
        assertEquals(ndArrayList,arrayAssertion);

        arrayAssertion.remove(2);
        ndArrayList.remove(2);
        assertEquals(ndArrayList,arrayAssertion);


        arrayAssertion.add(5,8.0);
        ndArrayList.add(5,8.0);
        assertEquals(arrayAssertion,ndArrayList);

        assertEquals(arrayAssertion.contains(8.0),ndArrayList.contains(8.0));
        assertEquals(arrayAssertion.indexOf(8.0),ndArrayList.indexOf(8.0));
        assertEquals(arrayAssertion.lastIndexOf(8.0),ndArrayList.lastIndexOf(8.0));
        assertEquals(ndArrayList.size(),ndArrayList.array().length());

    }
}
