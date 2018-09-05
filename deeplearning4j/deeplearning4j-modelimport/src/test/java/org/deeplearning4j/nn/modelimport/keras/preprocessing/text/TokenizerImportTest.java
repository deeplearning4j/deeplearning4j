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

package org.deeplearning4j.nn.modelimport.keras.preprocessing.text;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Import Keras Tokenizer
 *
 * @author Max Pumperla
 */
public class TokenizerImportTest {

    ClassLoader classLoader = getClass().getClassLoader();


    @Test
    public void importTest() throws IOException, InvalidKerasConfigurationException {

        String path = "modelimport/keras/preprocessing/tokenizer.json";

        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        KerasTokenizer tokenizer = KerasTokenizer.fromJson(configResource.getFile().getAbsolutePath());

        assertEquals(100, tokenizer.getNumWords().intValue());
        assertTrue(tokenizer.isLower());
        assertEquals(" ", tokenizer.getSplit());
        assertFalse(tokenizer.isCharLevel());
        assertEquals(0, tokenizer.getDocumentCount().intValue());


    }
}
