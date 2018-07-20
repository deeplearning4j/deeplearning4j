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

package org.ansj.util;

import org.ansj.domain.Term;
import org.ansj.domain.TermNatures;
import org.ansj.recognition.impl.NatureRecognition;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.util.WordAlert;

public class NameFix {
    /**
     * 人名消歧,比如.邓颖超生前->邓颖 超生 前 fix to 丁颖超 生 前! 规则的方式增加如果两个人名之间连接是- ， ·，•则连接
     */
    public static void nameAmbiguity(Term[] terms, Forest... forests) {
        Term from = null;
        Term term = null;
        Term next = null;
        for (int i = 0; i < terms.length - 1; i++) {
            term = terms[i];
            if (term != null && term.termNatures() == TermNatures.NR && term.getName().length() == 2) {
                next = terms[i + 2];
                if (next.termNatures().personAttr.split > 0) {
                    term.setName(term.getName() + next.getName().charAt(0));
                    terms[i + 2] = null;

                    String name = next.getName().substring(1);
                    terms[i + 3] = new Term(name, next.getOffe() + 1,
                                    new NatureRecognition(forests).getTermNatures(name));
                    TermUtil.termLink(term, terms[i + 3]);
                    TermUtil.termLink(terms[i + 3], next.to());
                }
            }
        }

        // 外国人名修正
        for (int i = 0; i < terms.length; i++) {
            term = terms[i];
            if (term != null && term.getName().length() == 1 && i > 0
                            && WordAlert.CharCover(term.getName().charAt(0)) == '·') {
                from = term.from();
                next = term.to();

                if (from.natrue().natureStr.startsWith("nr") && next.natrue().natureStr.startsWith("nr")) {
                    from.setName(from.getName() + term.getName() + next.getName());
                    TermUtil.termLink(from, next.to());
                    terms[i] = null;
                    terms[i + 1] = null;
                }
            }
        }

    }
}
