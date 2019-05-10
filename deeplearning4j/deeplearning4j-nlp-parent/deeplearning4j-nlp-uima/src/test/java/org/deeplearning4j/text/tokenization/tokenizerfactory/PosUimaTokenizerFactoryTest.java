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

package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;


/**
 * @author raver119@gmail.com
 */
public class PosUimaTokenizerFactoryTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testCreate1() throws Exception {
        String[] posTags = new String[] {"NN"};
        PosUimaTokenizerFactory factory = new PosUimaTokenizerFactory(Arrays.asList(posTags));
        Tokenizer tokenizer = factory.create("some test string");
        List<String> tokens = tokenizer.getTokens();
        System.out.println("Tokens: " + tokens);

        Assert.assertEquals(3, tokens.size());
        Assert.assertEquals("NONE", tokens.get(0));
        Assert.assertEquals("test", tokens.get(1));
        Assert.assertEquals("string", tokens.get(2));
    }

    @Test
    public void testCreate2() throws Exception {
        String[] posTags = new String[] {"NN"};
        PosUimaTokenizerFactory factory = new PosUimaTokenizerFactory(Arrays.asList(posTags), true);
        Tokenizer tokenizer = factory.create("some test string");
        List<String> tokens = tokenizer.getTokens();
        System.out.println("Tokens: " + tokens);

        Assert.assertEquals(2, tokens.size());
        Assert.assertEquals("test", tokens.get(0));
        Assert.assertEquals("string", tokens.get(1));
    }
}
