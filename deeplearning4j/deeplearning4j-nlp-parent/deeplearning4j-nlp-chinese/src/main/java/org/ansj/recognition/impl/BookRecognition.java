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

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * 基于规则的新词发现 jijiang feidiao
 * 
 * @author ansj
 * 
 */
public class BookRecognition implements Recognition {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    private static final Nature nature = new Nature("book");

    private static Map<String, String> ruleMap = new HashMap<>();

    static {
        ruleMap.put("《", "》");
    }

    @Override
    public void recognition(Result result) {
        List<Term> terms = result.getTerms();
        String end = null;
        String name;

        LinkedList<Term> mergeList = null;

        List<Term> list = new LinkedList<>();

        for (Term term : terms) {
            name = term.getName();
            if (end == null) {
                if ((end = ruleMap.get(name)) != null) {
                    mergeList = new LinkedList<>();
                    mergeList.add(term);
                } else {
                    list.add(term);
                }
            } else {
                mergeList.add(term);
                if (end.equals(name)) {

                    Term ft = mergeList.pollFirst();
                    for (Term sub : mergeList) {
                        ft.merage(sub);
                    }
                    ft.setNature(nature);
                    list.add(ft);
                    mergeList = null;
                    end = null;
                }
            }
        }

        if (mergeList != null) {
            for (Term term : list) {
                list.add(term);
            }
        }

        result.setTerms(list);
    }

}
