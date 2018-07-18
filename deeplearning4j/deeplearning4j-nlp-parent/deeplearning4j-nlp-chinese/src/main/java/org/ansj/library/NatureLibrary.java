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

package org.ansj.library;

import org.ansj.domain.Nature;
import org.ansj.domain.Term;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * 这里封装了词性和词性之间的关系.以及词性的索引.这是个好东西. 里面数组是从ict里面找来的. 不是很新.没有语料无法训练
 * 
 * @author ansj
 * 
 */
public class NatureLibrary {

    private static final Log logger = LogFactory.getLog(NatureLibrary.class);

    private static final int YI = 1;
    private static final int FYI = -1;
    /**
     * 词性的字符串对照索引位的hashmap(我发现我又效率狂了.不能这样啊)
     */
    private static final HashMap<String, Nature> NATUREMAP = new HashMap<>();

    /**
     * 词与词之间的关系.对照natureARRAY,natureMap
     */
    private static int[][] NATURETABLE = null;

    /**
     * 初始化对照表
     */
    static {
        init();
    }

    private static void init() {
        String split = "\t";
        int maxLength = 0;
        String temp = null;
        String[] strs = null;
        // 加载词对照性表
        try (BufferedReader reader = MyStaticValue.getNatureMapReader()) {
            int p0 = 0;
            int p1 = 0;
            int p2 = 0;
            while ((temp = reader.readLine()) != null) {
                strs = temp.split(split);
                if (strs.length != 4)
                    continue;

                p0 = Integer.parseInt(strs[0]);
                p1 = Integer.parseInt(strs[1]);
                p2 = Integer.parseInt(strs[3]);
                NATUREMAP.put(strs[2], new Nature(strs[2], p0, p1, p2));
                maxLength = Math.max(maxLength, p1);
            }
        } catch (IOException e) {
            logger.warn("词性列表加载失败!", e);
        }
        // 加载词性关系
        try (BufferedReader reader = MyStaticValue.getNatureTableReader()) {
            NATURETABLE = new int[maxLength + 1][maxLength + 1];
            int j = 0;
            while ((temp = reader.readLine()) != null) {
                if (StringUtil.isBlank(temp))
                    continue;
                strs = temp.split(split);
                for (int i = 0; i < strs.length; i++) {
                    NATURETABLE[j][i] = Integer.parseInt(strs[i]);
                }
                j++;
            }
        } catch (IOException e) {
            logger.warn("加载词性关系失败!", e);
        }
    }

    /**
     * 获得两个词性之间的频率
     * 
     * @param from
     * @param to
     * @return
     */
    public static int getTwoNatureFreq(Nature from, Nature to) {
        if (from.index < 0 || to.index < 0) {
            return 0;
        }
        return NATURETABLE[from.index][to.index];
    }

    /**
     * 获得两个term之间的频率
     * 
     * @param fromTerm
     * @param toTerm
     * @return
     */
    public static int getTwoTermFreq(Term fromTerm, Term toTerm) {
        Nature from = fromTerm.natrue();
        Nature to = toTerm.natrue();
        if (from.index < 0 || to.index < 0) {
            return 0;
        }
        return NATURETABLE[from.index][to.index];
    }

    /**
     * 根据字符串得道词性.没有就创建一个
     * 
     * @param natureStr
     * @return
     */
    public static Nature getNature(String natureStr) {
        Nature nature = NATUREMAP.get(natureStr);
        if (nature == null) {
            nature = new Nature(natureStr, FYI, FYI, YI);
            NATUREMAP.put(natureStr, nature);
            return nature;
        }
        return nature;
    }
}
