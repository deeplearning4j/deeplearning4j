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

package org.deeplearning4j.text.tokenization.tokenizer;

import org.deeplearning4j.text.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by kepricon on 16. 10. 24.
 */
public class KoreanTokenizerTest {
    @Test
    public void testKoreanTokenizer() throws Exception {
        String toTokenize = "세계 최초의 상용 수준 오픈소스 딥러닝 라이브러리입니다";
        TokenizerFactory t = new KoreanTokenizerFactory();
        Tokenizer tokenizer = t.create(toTokenize);
        String[] expect = {"세계", "최초", "의", "상용", "수준", "오픈소스", "딥", "러닝", "라이브러리", "입니", "다"};

        assertEquals(expect.length, tokenizer.countTokens());

        for (int i = 0; i < tokenizer.countTokens(); ++i) {
            assertEquals(tokenizer.nextToken(), expect[i]);
        }
    }

}
