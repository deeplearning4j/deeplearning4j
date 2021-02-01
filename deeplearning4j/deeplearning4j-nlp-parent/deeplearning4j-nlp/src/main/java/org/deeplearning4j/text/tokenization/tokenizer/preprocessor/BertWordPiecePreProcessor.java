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

package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

import java.text.Normalizer;
import java.util.List;
import java.util.Map;

/**
 * A preprocessor for cleaning/normaling text. Does the following:
 * 1. Optionally converts all characters to lower case
 * 2. Optionally strips accents off characters
 * 3. Strips all control characters
 * 4. Replaces whitespace characters with space ' ' (this includes newline and tab)
 * 5. Appends spaces before/after Chinese characters
 */
public class BertWordPiecePreProcessor implements TokenPreProcess {

    public static final char REPLACEMENT_CHAR = 0xfffd;

    protected final boolean lowerCase;
    protected final boolean stripAccents;
    protected final IntSet charSet;

    public BertWordPiecePreProcessor(){
        this(false, false, null);
    }

    /**
     *
     * @param lowerCase If true: tokenization should convert all characters to lower case
     * @param stripAccents  If true: strip accents off characters. Usually same as lower case. Should be true when using "uncased" official BERT TensorFlow models
     */
    public BertWordPiecePreProcessor(boolean lowerCase, boolean stripAccents, Map<String,Integer> vocab){
        this.lowerCase = lowerCase;
        this.stripAccents = stripAccents;
        if(vocab != null) {
            charSet = new IntOpenHashSet();
            for (String s : vocab.keySet()) {
                int cpNum = 0;
                int n = s.codePointCount(0, s.length());
                int charOffset = 0;
                while (cpNum++ < n) {
                    int cp = s.codePointAt(charOffset);
                    charOffset += Character.charCount(cp);
                    charSet.add(cp);
                }
            }
        } else {
            charSet = null;
        }
    }

    @Override
    public String preProcess(String token) {
        if(stripAccents) {
            token = Normalizer.normalize(token, Normalizer.Form.NFD);
        }

        int n = token.codePointCount(0, token.length());
        StringBuilder sb = new StringBuilder();
        int charOffset = 0;
        int cps = 0;
        while(cps++ < n){
            int cp = token.codePointAt(charOffset);
            charOffset += Character.charCount(cp);

            //Remove control characters and accents
            if(cp == 0 || cp == REPLACEMENT_CHAR || isControlCharacter(cp) || (stripAccents && Character.getType(cp) == Character.NON_SPACING_MARK))
                continue;

            //Convert to lower case if necessary
            if(lowerCase){
                cp = Character.toLowerCase(cp);
            }

            //Replace whitespace chars with space
            if(isWhiteSpace(cp)) {
                sb.append(' ');
                continue;
            }

            if(charSet != null && !charSet.contains(cp)){
                //Skip unknown character (out-of-vocab - though this should rarely happen)
                continue;
            }

            //Handle Chinese and other characters
            if(isChineseCharacter(cp)){
                sb.append(' ');
                sb.appendCodePoint(cp);
                sb.append(' ');
                continue;
            }

            //All other characters - keep
            sb.appendCodePoint(cp);
        }

        return sb.toString();
    }

    public static boolean isControlCharacter(int cp){
        //Treat newline/tab as whitespace
        if(cp == '\t' || cp == '\n' || cp == '\r')
            return false;
        int type = Character.getType(cp);
        return type == Character.CONTROL || type == Character.FORMAT;
    }

    public static boolean isWhiteSpace(int cp){
        //Treat newline/tab as whitespace
        if(cp == '\t' || cp == '\n' || cp == '\r')
            return true;
        int type = Character.getType(cp);
        return type == Character.SPACE_SEPARATOR;
    }

    public static boolean isChineseCharacter(int cp) {
        //Remove any CJK Unicode code block characters
        // https://en.wikipedia.org/wiki/List_of_CJK_Unified_Ideographs,_part_1_of_4
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
                (cp >= 0x3400 && cp <= 0x4DBF) ||
                (cp >= 0x20000 && cp <= 0x2A6DF) ||
                (cp >= 0x2A700 && cp <= 0x2B73F) ||
                (cp >= 0x2B740 && cp <= 0x2B81F) ||
                (cp >= 0x2B820 && cp <= 0x2CEAF) ||
                (cp >= 0xF900 && cp <= 0xFAFF) ||
                (cp >= 0x2F800 && cp <= 0x2FA1F);
    }


    /**
     * Reconstruct the String from tokens
     * @param tokens
     * @return
     */
    public static String reconstructFromTokens(List<String> tokens){
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(String s : tokens){
            if(s.startsWith("##")){
                sb.append(s.substring(2));
            } else {
                if(!first && !".".equals(s))
                    sb.append(" ");
                sb.append(s);
                first = false;
//            }
            }
        }
        return sb.toString();
    }
}
