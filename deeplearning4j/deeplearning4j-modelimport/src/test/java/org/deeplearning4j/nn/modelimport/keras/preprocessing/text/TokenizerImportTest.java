/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras.preprocessing.text;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.junit.Test;
import org.nd4j.common.resources.Resources;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 * Import Keras Tokenizer
 *
 * @author Max Pumperla
 */
public class TokenizerImportTest extends BaseDL4JTest {

    ClassLoader classLoader = getClass().getClassLoader();


    @Test(timeout=300000)
    public void importTest() throws IOException, InvalidKerasConfigurationException {

        String path = "modelimport/keras/preprocessing/tokenizer.json";

        KerasTokenizer tokenizer = KerasTokenizer.fromJson(Resources.asFile(path).getAbsolutePath());

        assertEquals(100, tokenizer.getNumWords().intValue());
        assertTrue(tokenizer.isLower());
        assertEquals(" ", tokenizer.getSplit());
        assertFalse(tokenizer.isCharLevel());
        assertEquals(0, tokenizer.getDocumentCount().intValue());


    }

    @Test(timeout=300000)
    public void importNumWordsNullTest() throws IOException, InvalidKerasConfigurationException {

        String path = "modelimport/keras/preprocessing/tokenizer_num_words_null.json";

        KerasTokenizer tokenizer = KerasTokenizer.fromJson(Resources.asFile(path).getAbsolutePath());

        assertNull(tokenizer.getNumWords());
        assertTrue(tokenizer.isLower());
        assertEquals(" ", tokenizer.getSplit());
        assertFalse(tokenizer.isCharLevel());
        assertEquals(0, tokenizer.getDocumentCount().intValue());
    }
}
