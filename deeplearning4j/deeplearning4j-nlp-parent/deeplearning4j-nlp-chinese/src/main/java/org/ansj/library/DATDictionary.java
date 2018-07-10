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

import org.ansj.dic.DicReader;
import org.ansj.domain.AnsjItem;
import org.ansj.domain.PersonNatureAttr;
import org.ansj.domain.TermNature;
import org.ansj.domain.TermNatures;
import org.ansj.library.name.PersonAttrLibrary;
import org.nlpcn.commons.lang.dat.DoubleArrayTire;
import org.nlpcn.commons.lang.dat.Item;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

public class DATDictionary {

    private static final Log LOG = LogFactory.getLog(DATDictionary.class);

    /**
     * 核心词典
     */
    private static final DoubleArrayTire DAT = loadDAT();

    /**
     * 数组长度
     */
    public static int arrayLength = DAT.arrayLength;

    /**
     * 加载词典
     * 
     * @return
     */
    private static DoubleArrayTire loadDAT() {
        long start = System.currentTimeMillis();
        try {
            DoubleArrayTire dat = DoubleArrayTire.loadText(DicReader.getInputStream("core.dic"), AnsjItem.class);
            // 人名识别必备的
            personNameFull(dat);
            // 记录词典中的词语，并且清除部分数据
            for (Item item : dat.getDAT()) {
                if (item == null || item.getName() == null) {
                    continue;
                }
                if (item.getStatus() < 2) {
                    item.setName(null);
                    continue;
                }
            }
            LOG.info("init core library ok use time : " + (System.currentTimeMillis() - start));
            return dat;
        } catch (InstantiationException e) {
            LOG.warn("无法实例化", e);
        } catch (IllegalAccessException e) {
            LOG.warn("非法访问", e);
        } catch (NumberFormatException e) {
            LOG.warn("数字格式异常", e);
        } catch (IOException e) {
            LOG.warn("IO异常", e);
        }

        return null;
    }

    private static void personNameFull(DoubleArrayTire dat) throws NumberFormatException, IOException {
        HashMap<String, PersonNatureAttr> personMap = new PersonAttrLibrary().getPersonMap();

        AnsjItem ansjItem = null;
        // 人名词性补录
        Set<Entry<String, PersonNatureAttr>> entrySet = personMap.entrySet();
        char c = 0;
        String temp = null;
        for (Entry<String, PersonNatureAttr> entry : entrySet) {
            temp = entry.getKey();

            if (temp.length() == 1 && (ansjItem = (AnsjItem) dat.getDAT()[temp.charAt(0)]) == null) {
                ansjItem = new AnsjItem();
                ansjItem.setBase(c);
                ansjItem.setCheck(-1);
                ansjItem.setStatus((byte) 3);
                ansjItem.setName(temp);
                dat.getDAT()[temp.charAt(0)] = ansjItem;
            } else {
                ansjItem = dat.getItem(temp);
            }

            if (ansjItem == null) {
                continue;
            }

            if ((ansjItem.termNatures) == null) {
                if (temp.length() == 1 && temp.charAt(0) < 256) {
                    ansjItem.termNatures = TermNatures.NULL;
                } else {
                    ansjItem.termNatures = new TermNatures(TermNature.NR);
                }
            }
            ansjItem.termNatures.setPersonNatureAttr(entry.getValue());
        }
    }

    public static int status(char c) {
        Item item = DAT.getDAT()[c];
        if (item == null) {
            return 0;
        }
        return item.getStatus();
    }

    /**
     * 判断一个词语是否在词典中
     * 
     * @param word
     * @return
     */
    public static boolean isInSystemDic(String word) {
        Item item = DAT.getItem(word);
        return item != null && item.getStatus() > 1;
    }

    public static AnsjItem getItem(int index) {
        AnsjItem item = DAT.getItem(index);
        if (item == null) {
            return AnsjItem.NULL;
        }

        return item;
    }

    public static AnsjItem getItem(String str) {
        AnsjItem item = DAT.getItem(str);
        if (item == null || item.getStatus() < 2) {
            return AnsjItem.NULL;
        }

        return item;
    }

    public static int getId(String str) {
        return DAT.getId(str);
    }

}
