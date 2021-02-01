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

package org.deeplearning4j.plot;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

import static org.junit.Assert.assertTrue;

public class Test6058 extends BaseDL4JTest {

    @Test
    public void test() throws Exception {
        //All zero input -> cosine similarity isn't defined
        //https://github.com/deeplearning4j/deeplearning4j/issues/6058
        val iterations = 10;
        val cacheList = new ArrayList<String>();

        int nWords  = 100;
        for(int i=0; i<nWords; i++ ) {
            cacheList.add("word_" + i);
        }

        //STEP 3: build a dual-tree tsne to use later
        System.out.println("Build model....");
        val tsne = new BarnesHutTsne.Builder()
                .setMaxIter(iterations)
                .theta(0.5)
                .normalize(false)
                .learningRate(1000)
                .useAdaGrad(false)
                //.usePca(false)
                .build();

        System.out.println("fit");
        INDArray weights = Nd4j.rand(new int[]{nWords, 100});
        weights.getRow(1).assign(0);
        try {
            tsne.fit(weights);
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("may not be defined"));
        }
    }

}
