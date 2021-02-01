/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package com.atilika.kuromoji.dict;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class InsertedDictionaryTest extends BaseDL4JTest {

    @Test
    public void testFeatureSize() {
        InsertedDictionary dictionary1 = new InsertedDictionary(9);
        InsertedDictionary dictionary2 = new InsertedDictionary(5);

        assertEquals("*,*,*,*,*,*,*,*,*", dictionary1.getAllFeatures(0));
        assertEquals("*,*,*,*,*", dictionary2.getAllFeatures(0));

        assertArrayEquals(new String[] {"*", "*", "*", "*", "*", "*", "*", "*", "*"},
                        dictionary1.getAllFeaturesArray(0));
        assertArrayEquals(new String[] {"*", "*", "*", "*", "*"}, dictionary2.getAllFeaturesArray(0));
    }
}
