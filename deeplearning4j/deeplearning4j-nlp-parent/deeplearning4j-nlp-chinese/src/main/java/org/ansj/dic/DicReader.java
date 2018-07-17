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

import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;

/**
 * 加载词典用的类
 * 
 * @author ansj
 */
public class DicReader {

    private static final Log logger = LogFactory.getLog();

    public static BufferedReader getReader(String name) {
        // maven工程修改词典加载方式
        InputStream in = DicReader.class.getResourceAsStream("/" + name);
        try {
            return new BufferedReader(new InputStreamReader(in, "UTF-8"));
        } catch (UnsupportedEncodingException e) {
            logger.warn("不支持的编码", e);
        }
        return null;
    }

    public static InputStream getInputStream(String name) {
        // maven工程修改词典加载方式
        InputStream in = DicReader.class.getResourceAsStream("/" + name);
        return in;
    }
}
