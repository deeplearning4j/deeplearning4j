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

package org.ansj.library.name;

import org.ansj.domain.PersonNatureAttr;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * 人名标注所用的词典就是简单的hashmap简单方便谁用谁知道,只在加载词典的时候用
 * 
 * @author ansj
 */

public class PersonAttrLibrary {

    private static final Log logger = LogFactory.getLog();

    private HashMap<String, PersonNatureAttr> pnMap = null;

    public PersonAttrLibrary() {}

    public HashMap<String, PersonNatureAttr> getPersonMap() {
        if (pnMap != null) {
            return pnMap;
        }
        init1();
        init2();
        return pnMap;
    }

    // name_freq
    private void init2() {
        Map<String, int[][]> personFreqMap = MyStaticValue.getPersonFreqMap();
        Set<Entry<String, int[][]>> entrySet = personFreqMap.entrySet();
        PersonNatureAttr pna = null;
        for (Entry<String, int[][]> entry : entrySet) {
            pna = pnMap.get(entry.getKey());
            if (pna == null) {
                pna = new PersonNatureAttr();
                pna.setlocFreq(entry.getValue());
                pnMap.put(entry.getKey(), pna);
            } else {
                pna.setlocFreq(entry.getValue());
            }

        }
    }

    // person.dic
    private void init1() {
        try (BufferedReader br = MyStaticValue.getPersonReader()) {
            pnMap = new HashMap<>();
            String temp = null;
            String[] strs = null;
            PersonNatureAttr pna = null;
            while ((temp = br.readLine()) != null) {
                pna = new PersonNatureAttr();
                strs = temp.split("\t");
                pna = pnMap.get(strs[0]);
                if (pna == null) {
                    pna = new PersonNatureAttr();
                }
                pna.addFreq(Integer.parseInt(strs[1]), Integer.parseInt(strs[2]));
                pnMap.put(strs[0], pna);
            }
        } catch (NumberFormatException e) {
            logger.warn("数字格式不正确", e);
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
    }
}
