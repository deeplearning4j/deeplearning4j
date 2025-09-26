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

package org.deeplearning4j.text.inputsanitation;

import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.List;

public class InputHomogenization {
    private String input;
    private List<String> ignoreCharactersContaining;
    private boolean preserveCase;

    /**
     * Input text to applyTransformToOrigin
     * @param input the input text to applyTransformToOrigin,
     * equivalent to calling this(input,false)
     * wrt preserving case
     */
    public InputHomogenization(String input) {
        this(input, false);
    }

    /**
     * 
     * @param input the input to applyTransformToOrigin
     * @param preserveCase whether to preserve case
     */
    public InputHomogenization(String input, boolean preserveCase) {
        this.input = input;
        this.preserveCase = preserveCase;
    }

    /**
     * 
     * @param input the input to applyTransformToOrigin
     * @param ignoreCharactersContaining ignore transformation of words
     * containigng specified strings
     */
    public InputHomogenization(String input, List<String> ignoreCharactersContaining) {
        this.input = input;
        this.ignoreCharactersContaining = ignoreCharactersContaining;
    }

    /**
     * Returns the normalized text passed in via constructor
     * @return the normalized text passed in via constructor
     */
    public String transform() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < input.length(); i++) {
            if (ignoreCharactersContaining != null
                            && ignoreCharactersContaining.contains(String.valueOf(input.charAt(i))))
                sb.append(input.charAt(i));
            else if (Character.isDigit(input.charAt(i)))
                sb.append("d");
            else if (Character.isUpperCase(input.charAt(i)) && !preserveCase)
                sb.append(Character.toLowerCase(input.charAt(i)));
            else
                sb.append(input.charAt(i));

        }

        String normalized = Normalizer.normalize(sb.toString(), Form.NFD);
        normalized = normalized.replace(".", "");
        normalized = normalized.replace(",", "");
        normalized = normalized.replaceAll("\"", "");
        normalized = normalized.replace("'", "");
        normalized = normalized.replace("(", "");
        normalized = normalized.replace(")", "");
        normalized = normalized.replace("“", "");
        normalized = normalized.replace("”", "");
        normalized = normalized.replace("…", "");
        normalized = normalized.replace("|", "");
        normalized = normalized.replace("/", "");
        normalized = normalized.replace("\\", "");
        normalized = normalized.replace("[", "");
        normalized = normalized.replace("]", "");
        normalized = normalized.replace("‘", "");
        normalized = normalized.replace("’", "");
        normalized = normalized.replaceAll("[!]+", "!");
        return normalized;
    }

}
