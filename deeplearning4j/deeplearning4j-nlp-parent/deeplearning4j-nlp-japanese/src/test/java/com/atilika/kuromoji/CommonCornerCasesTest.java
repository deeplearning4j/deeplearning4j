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

package com.atilika.kuromoji;

import org.deeplearning4j.BaseDL4JTest;

import java.util.Arrays;

import static com.atilika.kuromoji.TestUtils.assertTokenSurfacesEquals;

public class CommonCornerCasesTest extends BaseDL4JTest {

    public static void testPunctuation(TokenizerBase tokenizer) {
        String gerryNoHanaNoHanashi = "僕の鼻はちょっと\r\n長いよ。";

        assertTokenSurfacesEquals(Arrays.asList("僕", "の", "鼻", "は", "ちょっと", "\r", "\n", "長い", "よ", "。"),

                        tokenizer.tokenize(gerryNoHanaNoHanashi));
    }
}
