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

import org.ansj.app.crf.Model;
import org.ansj.app.crf.SplitWord;
import org.ansj.app.crf.model.CRFModel;
import org.ansj.dic.PathToStream;
import org.ansj.domain.KV;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.logging.Log;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class CrfLibrary {

    private static final Log LOG = MyStaticValue.getLog(CrfLibrary.class);

    // CRF模型
    private static final Map<String, KV<String, SplitWord>> CRF = new HashMap<>();

    public static final String DEFAULT = "crf";

    static {
        for (Entry<String, String> entry : MyStaticValue.ENV.entrySet()) {
            if (entry.getKey().startsWith(DEFAULT)) {
                put(entry.getKey(), entry.getValue());
            }
        }
        putIfAbsent(DEFAULT, "jar://crf.model");
    }

    public static SplitWord get() {
        return get(DEFAULT);
    }

    /**
     * 根据key获取crf分词器
     * 
     * @param key
     * @return crf分词器
     */
    public static SplitWord get(String key) {

        KV<String, SplitWord> kv = CRF.get(key);

        if (kv == null) {
            if (MyStaticValue.ENV.containsKey(key)) {
                putIfAbsent(key, MyStaticValue.ENV.get(key));
                return get(key);
            }
            LOG.warn("crf " + key + " not found in config ");
            return null;
        }

        SplitWord sw = kv.getV();
        if (sw == null) {
            sw = initCRFModel(kv);
        }
        return sw;
    }

    /**
     * 加载CRF模型
     * 
     * @param modelPath
     * @return
     */
    private static synchronized SplitWord initCRFModel(KV<String, SplitWord> kv) {
        try {
            if (kv.getV() != null) {
                return kv.getV();
            }

            long start = System.currentTimeMillis();
            LOG.debug("begin init crf model!");
            try (InputStream is = PathToStream.stream(kv.getK())) {
                SplitWord crfSplitWord = new SplitWord(Model.load(CRFModel.class, is));
                kv.setV(crfSplitWord);
                LOG.info("load crf use time:" + (System.currentTimeMillis() - start) + " path is : " + kv.getK());
                return crfSplitWord;
            }
        } catch (Exception e) {
            LOG.error(kv + " load err " + e.getMessage());
            return null;
        }
    }

    /**
     * 动态添加
     * 
     * @param dicDefault
     * @param dicDefault2
     * @param dic2
     */
    public static void put(String key, String path) {

        put(key, path, null);
    }

    public static void put(String key, String path, SplitWord sw) {
        CRF.put(key, KV.with(path, sw));
        MyStaticValue.ENV.put(key, path);
    }

    /**
     * 删除一个key
     * 
     * @param key
     * @return
     */
    public static KV<String, SplitWord> remove(String key) {
        MyStaticValue.ENV.remove(key);
        return CRF.remove(key);
    }

    /**
     * 刷新一个,将值设置为null
     * 
     * @param key
     * @return
     */
    public static void reload(String key) {
        KV<String, SplitWord> kv = CRF.get(key);
        if (kv != null) {
            CRF.get(key).setV(null);
        }

        LOG.warn("make sure ,this reload not use same obj , it to instance a new model");
    }

    public static Set<String> keys() {
        return CRF.keySet();
    }

    public static void putIfAbsent(String key, String path) {
        if (!CRF.containsKey(key)) {
            CRF.put(key, KV.with(path, (SplitWord) null));
        }
    }
}
