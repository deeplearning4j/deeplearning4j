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

package org.deeplearning4j.text.tokenization.tokenizer;


import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A tokenizer that works with a vocab from a published bert model and tokenizes a token at a time from a stream
 * @author Paul Dubs
 */
@Slf4j
public class BertWordPieceStreamTokenizer extends BertWordPieceTokenizer {


    public BertWordPieceStreamTokenizer(InputStream tokens, Charset encoding, NavigableMap<String, Integer> vocab, TokenPreProcess preTokenizePreProcessor, TokenPreProcess tokenPreProcess) {
        super(readAndClose(tokens, encoding), vocab, preTokenizePreProcessor, tokenPreProcess);
    }


    public static String readAndClose(InputStream is, Charset encoding){
        try {
            return IOUtils.toString(is, encoding);
        } catch (IOException e){
            throw new RuntimeException("Error reading from stream", e);
        } finally {
            IOUtils.closeQuietly(is);
        }
    }
}
