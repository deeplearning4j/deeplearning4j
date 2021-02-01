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
package com.atilika.kuromoji.ipadic.compile;

import com.atilika.kuromoji.compile.DictionaryCompilerBase;
import com.atilika.kuromoji.compile.TokenInfoDictionaryCompilerBase;

import java.io.IOException;

public class DictionaryCompiler extends DictionaryCompilerBase {

    @Override
    protected TokenInfoDictionaryCompilerBase getTokenInfoDictionaryCompiler(String encoding) {
        return new TokenInfoDictionaryCompiler(encoding);
    }

    public static void main(String[] args) throws IOException {
        DictionaryCompiler dictionaryBuilder = new DictionaryCompiler();
        dictionaryBuilder.build(args);
    }
}
