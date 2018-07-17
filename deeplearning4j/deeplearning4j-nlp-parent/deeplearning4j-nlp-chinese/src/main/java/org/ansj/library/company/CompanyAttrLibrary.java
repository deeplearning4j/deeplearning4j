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

package org.ansj.library.company;

import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * 机构名识别词典加载类
 * 
 * @author ansj
 * 
 */
public class CompanyAttrLibrary {

    private static final Log logger = LogFactory.getLog();

    private static HashMap<String, int[]> cnMap = null;

    private CompanyAttrLibrary() {}

    public static HashMap<String, int[]> getCompanyMap() {
        if (cnMap != null) {
            return cnMap;
        }
        init();
        return cnMap;
    }

    // company_freq

    private static void init() {
        try (BufferedReader br = MyStaticValue.getCompanReader()) {
            cnMap = new HashMap<>();
            String temp = null;
            String[] strs = null;
            int[] cna = null;
            while ((temp = br.readLine()) != null) {
                strs = temp.split("\t");
                cna = new int[2];
                cna[0] = Integer.parseInt(strs[1]);
                cna[1] = Integer.parseInt(strs[2]);
                cnMap.put(strs[0], cna);
            }
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
    }

}
