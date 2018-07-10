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

package org.ansj.app.crf;

import org.ansj.app.crf.model.CRFModel;
import org.ansj.app.crf.model.CRFppTxtModel;
import org.ansj.app.crf.model.WapitiCRFModel;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.MapCount;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.*;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.GZIPOutputStream;

public abstract class Model {

    public static final Log logger = LogFactory.getLog(Model.class);

    protected Config config;

    protected SmartForest<float[]> featureTree = null;

    protected float[][] status = new float[Config.TAG_NUM][Config.TAG_NUM];

    public int allFeatureCount = 0;

    /**
     * 判断当前数据流是否是本实例
     * 
     * @param is
     * @return
     */
    public abstract boolean checkModel(String modelPath) throws IOException;

    /**
     * 模型读取
     * 
     * @param path
     * @return
     * @return
     * @throws Exception
     */
    public static Model load(String modelPath) throws Exception {
        Model model = new CRFModel();
        if (model.checkModel(modelPath)) {
            return model.loadModel(modelPath);
        }
        model = new CRFppTxtModel();

        if (model.checkModel(modelPath)) {
            return model.loadModel(modelPath);
        }
        model = new WapitiCRFModel();
        if (model.checkModel(modelPath)) {
            return model.loadModel(modelPath);
        }
        throw new Exception("I did not know what type of model by file " + modelPath);
    }

    /**
     * 模型读取
     * 
     * @param path
     * @return
     * @return
     * @throws Exception
     */
    public static Model load(Class<? extends Model> c, InputStream is) throws Exception {
        Model model = c.newInstance();
        return model.loadModel(is);
    }

    /**
     * 不同的模型实现自己的加载模型类
     * 
     * @throws Exception
     */
    public abstract Model loadModel(String modelPath) throws Exception;

    public abstract Model loadModel(InputStream is) throws Exception;

    /**
     * 获得特征所在权重数组
     * 
     * @param featureStr
     * @return
     */
    public float[] getFeature(char... chars) {
        if (chars == null) {
            return null;
        }
        SmartForest<float[]> sf = featureTree;
        sf = sf.getBranch(chars);
        if (sf == null || sf.getParam() == null) {
            return null;
        }
        return sf.getParam();
    }

    public Config getConfig() {
        return this.config;
    }

    /**
     * tag转移率
     * 
     * @param s1
     * @param s2
     * @return
     */
    public float tagRate(int s1, int s2) {
        return status[s1][s2];
    }

    /**
     * 增加特征到特征数中
     * 
     * @param cs
     * @param tempW
     */
    protected static void printFeatureTree(String cs, float[] tempW) {
        String name = "*";
        if (tempW.length == 4) {
            name = "U";
        }
        name += "*" + (cs.charAt(cs.length() - 1) - Config.FEATURE_BEGIN + 1) + ":" + cs.substring(0, cs.length() - 1);
        for (int i = 0; i < tempW.length; i++) {
            if (tempW[i] != 0) {
                System.out.println(name + "\t" + Config.getTagName(i / 4 - 1) + "\t" + Config.getTagName(i % 4) + "\t"
                                + tempW[i]);
            }

        }
    }

    /**
     * 将model序列化到硬盘
     * 
     * @param path
     * @throws IOException
     * @throws FileNotFoundException
     */
    public void writeModel(String path) {
        try (FileOutputStream fso = new FileOutputStream(path)) {
            ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(fso));
            oos.writeUTF(CRFModel.VERSION);
            oos.writeObject(status);
            oos.writeObject(config.getTemplate());
            Map<String, float[]> map = featureTree.toMap();
            MapCount<Integer> mc = new MapCount<>();
            for (float[] v : map.values()) {
                mc.add(v.length);
            }
            for (Entry<Integer, Double> entry : mc.get().entrySet()) {
                int win = entry.getKey();
                oos.writeInt(win);// 宽度
                oos.writeInt(entry.getValue().intValue());// 个数
                for (Entry<String, float[]> e : map.entrySet()) {
                    if (e.getValue().length == win) {
                        oos.writeUTF(e.getKey());
                        float[] value = e.getValue();
                        for (int i = 0; i < win; i++) {
                            oos.writeFloat(value[i]);
                        }
                    }
                }
            }
            oos.writeInt(0);
            oos.writeInt(0);
            oos.flush();
        } catch (FileNotFoundException e) {
            logger.warn("文件没有找到", e);
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
    }
}
