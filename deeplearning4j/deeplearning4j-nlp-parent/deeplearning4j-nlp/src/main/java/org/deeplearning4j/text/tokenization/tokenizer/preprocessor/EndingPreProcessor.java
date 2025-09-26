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

package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

/**
 * Gets rid of endings:
 *
 *    ed,ing, ly, s, .
 * @author Adam Gibson
 */
public class EndingPreProcessor implements TokenPreProcess {
    @Override
    public String preProcess(String token) {
        if (token.endsWith("s") && !token.endsWith("ss"))
            token = token.substring(0, token.length() - 1);
        if (token.endsWith("."))
            token = token.substring(0, token.length() - 1);
        if (token.endsWith("ed"))
            token = token.substring(0, token.length() - 2);
        if (token.endsWith("ing"))
            token = token.substring(0, token.length() - 3);
        if (token.endsWith("ly"))
            token = token.substring(0, token.length() - 2);
        return token;
    }
}
