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

package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

/**
 * A ToeknPreProcess implementation that removes puncuation marks and lower-cases.
 * <br>
 * Note that the implementation uses String#toLowerCase(String) and its behavior depends on the default locale.
 * @see StringCleaning#stripPunct(String)
 * @author jeffreytang
 */
public class CommonPreprocessor implements TokenPreProcess {
    @Override
    public String preProcess(String token) {
        return StringCleaning.stripPunct(token).toLowerCase();
    }
}
