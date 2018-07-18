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

package org.datavec.nlp.tokenization.tokenizer;

import org.apache.uima.cas.CAS;
import org.apache.uima.fit.util.JCasUtil;
import org.cleartk.token.type.Token;
import org.datavec.nlp.uima.UimaResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Tokenizer based on the passed in analysis engine
 * @author Adam Gibson
 *
 */
public class UimaTokenizer implements Tokenizer {

    private List<String> tokens;
    private int index;
    private static Logger log = LoggerFactory.getLogger(UimaTokenizer.class);
    private boolean checkForLabel;
    private TokenPreProcess tokenPreProcessor;


    public UimaTokenizer(String tokens, UimaResource resource, boolean checkForLabel) {

        this.checkForLabel = checkForLabel;
        this.tokens = new ArrayList<>();
        try {
            CAS cas = resource.process(tokens);

            Collection<Token> tokenList = JCasUtil.select(cas.getJCas(), Token.class);

            for (Token t : tokenList) {

                if (!checkForLabel || valid(t.getCoveredText()))
                    if (t.getLemma() != null)
                        this.tokens.add(t.getLemma());
                    else if (t.getStem() != null)
                        this.tokens.add(t.getStem());
                    else
                        this.tokens.add(t.getCoveredText());
            }


            resource.release(cas);


        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }

    }

    private boolean valid(String check) {
        if (check.matches("<[A-Z]+>") || check.matches("</[A-Z]+>"))
            return false;
        return true;
    }



    @Override
    public boolean hasMoreTokens() {
        return index < tokens.size();
    }

    @Override
    public int countTokens() {
        return tokens.size();
    }

    @Override
    public String nextToken() {
        String ret = tokens.get(index);
        index++;
        if (tokenPreProcessor != null) {
            ret = tokenPreProcessor.preProcess(ret);
        }
        return ret;
    }

    @Override
    public List<String> getTokens() {
        List<String> tokens = new ArrayList<>();
        while (hasMoreTokens()) {
            tokens.add(nextToken());
        }
        return tokens;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcessor = tokenPreProcessor;
    }



}
