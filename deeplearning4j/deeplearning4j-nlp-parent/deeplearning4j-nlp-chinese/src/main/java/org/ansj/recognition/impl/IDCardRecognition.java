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

package org.ansj.recognition.impl;

import org.ansj.domain.Nature;
import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.recognition.Recognition;

import java.util.Iterator;
import java.util.List;

/**
 * 基于规则的新词发现，身份证号码识别
 * 
 * @author ansj
 * 
 */
public class IDCardRecognition implements Recognition {
    /**
     * 
     */
    private static final long serialVersionUID = -32133440735240290L;
    private static final Nature ID_CARD_NATURE = new Nature("idcard");

    @Override
    public void recognition(Result result) {

        List<Term> terms = result.getTerms();

        for (Term term : terms) {
            if ("m".equals(term.getNatureStr())) {

                if (term.getName().length() == 18) {
                    term.setNature(ID_CARD_NATURE);
                } else if (term.getName().length() == 17) {
                    Term to = term.to();
                    if ("x".equals(to.getName())) {
                        term.merage(to);
                        to.setName(null);
                        term.setNature(ID_CARD_NATURE);
                    }
                }

            }
        }

        for (Iterator<Term> iterator = terms.iterator(); iterator.hasNext();) {
            Term term = iterator.next();
            if (term.getName() == null) {
                iterator.remove();
            }
        }

    }

}
