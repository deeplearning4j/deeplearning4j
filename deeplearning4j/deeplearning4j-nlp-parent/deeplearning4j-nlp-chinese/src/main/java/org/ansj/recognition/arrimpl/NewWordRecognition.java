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

package org.ansj.recognition.arrimpl;

import org.ansj.dic.LearnTool;
import org.ansj.domain.Nature;
import org.ansj.domain.NewWord;
import org.ansj.domain.Term;
import org.ansj.util.TermUtil;
import org.ansj.util.TermUtil.InsertTermType;
import org.nlpcn.commons.lang.tire.domain.SmartForest;

/**
 * 新词识别
 * 
 * @author ansj
 * 
 */
public class NewWordRecognition {

    private Term[] terms = null;

    private double score;

    private StringBuilder sb = new StringBuilder();

    private SmartForest<NewWord> forest = null;

    private SmartForest<NewWord> branch = null;

    // private int offe = -1;
    // private int endOffe = -1;
    private Nature tempNature;

    private Term from;

    private Term to;

    // 偏移量
    private int offe;

    public NewWordRecognition(LearnTool learn) {
        forest = learn.getForest();
        branch = learn.getForest();
    }

    public void recognition(Term[] terms) {
        this.terms = terms;
        if (branch == null) {
            return;
        }
        int length = terms.length - 1;

        Term term = null;
        for (int i = 0; i < length; i++) {
            if (terms[i] == null) {
                continue;
            } else {
                from = terms[i].from();
                terms[i].score(0);
                terms[i].selfScore(0);
            }

            branch = branch.getBranch(terms[i].getName());

            if (branch == null || branch.getStatus() == 3) {
                reset();
                continue;
            }

            offe = i;

            // 循环查找添加
            term = terms[i];
            sb.append(term.getName());
            if (branch.getStatus() == 2) {
                term.selfScore(branch.getParam().getScore());
            }
            boolean flag = true;
            while (flag) {
                term = term.to();
                branch = branch.getBranch(term.getName());
                // 如果没有找到跳出
                if (branch == null) {
                    break;
                }

                switch (branch.getStatus()) {
                    case 1:
                        sb.append(term.getName());
                        continue;
                    case 2:
                        sb.append(term.getName());
                        score = branch.getParam().getScore();
                        tempNature = branch.getParam().getNature();
                        to = term.to();
                        makeNewTerm();
                        continue;
                    case 3:
                        sb.append(term.getName());
                        score = branch.getParam().getScore();
                        tempNature = branch.getParam().getNature();
                        to = term.to();
                        makeNewTerm();
                        flag = false;
                        break;
                    default:
                        System.out.println("怎么能出现0呢?");
                        break;
                }
            }
            reset();
        }
    }

    private void makeNewTerm() {
        Term term = new Term(sb.toString(), offe, tempNature.natureStr, 1);
        term.selfScore(score);
        term.setNature(tempNature);
        if (sb.length() > 3) {
            term.setSubTerm(TermUtil.getSubTerm(from, to));
        }
        TermUtil.termLink(from, term);
        TermUtil.termLink(term, to);
        TermUtil.insertTerm(terms, term, InsertTermType.SCORE_ADD_SORT);
        TermUtil.parseNature(term);
    }

    /**
     * 重置
     */
    private void reset() {
        offe = -1;
        tempNature = null;
        branch = forest;
        score = 0;
        sb = new StringBuilder();
    }

}
