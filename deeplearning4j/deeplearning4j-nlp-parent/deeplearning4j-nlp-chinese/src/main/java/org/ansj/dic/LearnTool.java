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

package org.ansj.dic;

import org.ansj.app.crf.SplitWord;
import org.ansj.domain.Nature;
import org.ansj.domain.NewWord;
import org.ansj.domain.TermNatures;
import org.ansj.recognition.arrimpl.AsianPersonRecognition;
import org.ansj.recognition.arrimpl.ForeignPersonRecognition;
import org.ansj.recognition.impl.NatureRecognition;
import org.ansj.util.Graph;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.CollectionUtil;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

/**
 * 新词发现,这是个线程安全的.所以可以多个对象公用一个
 * 
 * @author ansj
 * 
 */
public class LearnTool {

    private SplitWord splitWord = null;

    /**
     * 是否开启学习机
     */
    public boolean isAsianName = true;

    public boolean isForeignName = true;

    /**
     * 告诉大家你学习了多少个词了
     */
    public int count;

    /**
     * 新词发现的结果集.可以序列化到硬盘.然后可以当做训练集来做.
     */
    private final SmartForest<NewWord> sf = new SmartForest<>();

    /**
     * 学习新词排除用户自定义词典那中的词语
     */
    private Forest[] forests;

    /**
     * 公司名称学习.
     * 
     * @param graph
     */
    public void learn(Graph graph, SplitWord splitWord, Forest... forests) {

        this.splitWord = splitWord;

        this.forests = forests;

        // 亚洲人名识别
        if (isAsianName) {
            findAsianPerson(graph);
        }

        // 外国人名识别
        if (isForeignName) {
            findForeignPerson(graph);
        }

    }

    private void findAsianPerson(Graph graph) {
        List<NewWord> newWords = new AsianPersonRecognition().getNewWords(graph.terms);
        addListToTerm(newWords);
    }

    private void findForeignPerson(Graph graph) {
        List<NewWord> newWords = new ForeignPersonRecognition().getNewWords(graph.terms);
        addListToTerm(newWords);
    }

    // 批量将新词加入到词典中
    private void addListToTerm(List<NewWord> newWords) {
        if (newWords.isEmpty())
            return;
        for (NewWord newWord : newWords) {

            TermNatures termNatures = new NatureRecognition(forests).getTermNatures(newWord.getName());

            if (termNatures == TermNatures.NULL) {
                addTerm(newWord);
            }
        }
    }

    /**
     * 增加一个新词到树中
     * 
     * @param newWord
     */
    public void addTerm(NewWord newWord) {
        NewWord temp = null;
        SmartForest<NewWord> smartForest = null;
        if ((smartForest = sf.getBranch(newWord.getName())) != null && smartForest.getParam() != null) {
            temp = smartForest.getParam();
            temp.update(newWord.getNature(), newWord.getAllFreq());
        } else {
            count++;
            if (splitWord == null) {
                newWord.setScore(-1);
            } else {
                newWord.setScore(-splitWord.cohesion(newWord.getName()));
            }

            synchronized (sf) {
                sf.add(newWord.getName(), newWord);
            }
        }
    }

    public SmartForest<NewWord> getForest() {
        return this.sf;
    }

    /**
     * 返回学习到的新词.
     * 
     * @param num 返回数目.0为全部返回
     * @return
     */
    public List<Entry<String, Double>> getTopTree(int num) {
        return getTopTree(num, null);
    }

    public List<Entry<String, Double>> getTopTree(int num, Nature nature) {
        if (sf.branches == null) {
            return null;
        }
        HashMap<String, Double> hm = new HashMap<>();
        for (int i = 0; i < sf.branches.length; i++) {
            valueResult(sf.branches[i], hm, nature);
        }
        List<Entry<String, Double>> sortMapByValue = CollectionUtil.sortMapByValue(hm, -1);
        if (num == 0) {
            return sortMapByValue;
        } else {
            num = Math.min(num, sortMapByValue.size());
            return sortMapByValue.subList(0, num);
        }
    }

    private void valueResult(SmartForest<NewWord> smartForest, HashMap<String, Double> hm, Nature nature) {

        if (smartForest == null || smartForest.branches == null) {
            return;
        }
        for (int i = 0; i < smartForest.branches.length; i++) {
            NewWord param = smartForest.branches[i].getParam();
            if (smartForest.branches[i].getStatus() == 3) {
                if (param.isActive() && (nature == null || param.getNature().equals(nature))) {
                    hm.put(param.getName(), param.getScore());
                }
            } else if (smartForest.branches[i].getStatus() == 2) {
                if (param.isActive() && (nature == null || param.getNature().equals(nature))) {
                    hm.put(param.getName(), param.getScore());
                }
                valueResult(smartForest.branches[i], hm, nature);
            } else {
                valueResult(smartForest.branches[i], hm, nature);
            }
        }
    }

    /**
     * 尝试激活，新词
     * 
     * @param name
     */
    public void active(String name) {
        SmartForest<NewWord> branch = sf.getBranch(name);
        if (branch != null && branch.getParam() != null) {
            branch.getParam().setActive(true);
        }
    }
}
