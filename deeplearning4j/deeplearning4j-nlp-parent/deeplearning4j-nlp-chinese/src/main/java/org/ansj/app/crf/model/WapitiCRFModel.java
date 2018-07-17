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

package org.ansj.app.crf.model;

import org.ansj.app.crf.Config;
import org.ansj.app.crf.Model;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.ObjConver;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.tuples.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * 加载wapiti生成的crf模型,测试使用的wapiti版本为:Wapiti v1.5.0
 * 
 * wapiti 下载地址:https://wapiti.limsi.fr/#download 在这里感谢作者所做的工作.
 * 
 * @author Ansj
 *
 */
public class WapitiCRFModel extends Model {

    @Override
    public WapitiCRFModel loadModel(String modelPath) throws Exception {
        try (InputStream is = IOUtil.getInputStream(modelPath)) {
            return loadModel(is);
        }
    }

    @Override
    public WapitiCRFModel loadModel(InputStream is) throws Exception {
        BufferedReader br = IOUtil.getReader(is, IOUtil.UTF8);

        long start = System.currentTimeMillis();

        logger.info("load wapiti model begin!");

        String temp = br.readLine();

        logger.info(temp); // #mdl#2#123

        Map<String, Integer> featureIndex = loadConfig(br);

        StringBuilder sb = new StringBuilder();
        for (int[] t1 : config.getTemplate()) {
            sb.append(Arrays.toString(t1) + " ");
        }

        logger.info("featureIndex is " + featureIndex);
        logger.info("load template ok template : " + sb);

        int[] statusCoven = loadTagCoven(br);

        List<Pair<String, String>> loadFeatureName = loadFeatureName(featureIndex, br);

        logger.info("load feature ok feature size : " + loadFeatureName.size());

        featureTree = new SmartForest<float[]>();

        loadFeatureWeight(br, statusCoven, loadFeatureName);

        logger.info("load wapiti model ok ! use time :" + (System.currentTimeMillis() - start));
        return this;
    }

    /**
     * 加载特征权重
     * 
     * @param br
     * @param featureNames
     * @param statusCoven
     * @throws Exception
     */
    private void loadFeatureWeight(BufferedReader br, int[] statusCoven, List<Pair<String, String>> featureNames)
                    throws Exception {

        int key = 0;

        int offe = 0;

        int tag = 0; // 赏析按标签为用来转换

        int len = 0; // 权重数组的大小

        int min, max = 0; // 设置边界

        String name = null; // 特征名称

        float[] tempW = null; // 每一个特征的权重

        String temp = br.readLine();

        for (Pair<String, String> pair : featureNames) {

            if (temp == null) {
                logger.warn(pair.getValue0() + "\t" + pair.getValue1() + " not have any weight ,so skip it !");
                continue;
            }

            char fc = Character.toUpperCase(pair.getValue0().charAt(0));

            len = fc == 'B' ? Config.TAG_NUM * Config.TAG_NUM
                            : fc == 'U' ? Config.TAG_NUM
                                            : fc == '*' ? (Config.TAG_NUM + Config.TAG_NUM * Config.TAG_NUM) : 0;

            if (len == 0) {
                throw new Exception("unknow feature type " + pair.getValue0());
            }

            min = max;
            max += len;
            if (fc == 'B') { // 特殊处理转换特征数组
                for (int i = 0; i < len; i++) {
                    String[] split = temp.split("=");
                    int from = statusCoven[i / Config.TAG_NUM];
                    int to = statusCoven[i % Config.TAG_NUM];
                    status[from][to] = ObjConver.getFloatValue(split[1]);
                    temp = br.readLine();
                }
            } else {

                name = pair.getValue1();

                tempW = new float[len];

                do {
                    String[] split = temp.split("=");

                    key = ObjConver.getIntValue(split[0]);

                    if (key >= max) { // 如果超过边界那么跳出
                        break;
                    }

                    offe = key - min;

                    tag = statusCoven[offe];

                    tempW[tag] = ObjConver.getFloatValue(split[1]);

                } while ((temp = br.readLine()) != null);

                this.featureTree.add(name, tempW); // 将特征增加到特征🌲中

                // printFeatureTree(name, tempW);
            }

        }

    }

    /**
     * 加载特征值 //11:*6:_x-1/的,
     * 
     * @param featureIndex
     * 
     * @param br
     * @return
     * @throws Exception
     */

    private List<Pair<String, String>> loadFeatureName(Map<String, Integer> featureIndex, BufferedReader br)
                    throws Exception {
        String temp = br.readLine();// #qrk#num
        int featureNum = ObjConver.getIntValue(StringUtil.matcherFirst("\\d+", temp)); // 找到特征个数

        List<Pair<String, String>> featureNames = new ArrayList<>();

        for (int i = 0; i < featureNum; i++) {
            temp = br.readLine();

            String[] split = temp.split(":");

            if (split.length == 2) {
                featureNames.add(Pair.with(split[1], ""));
                continue;
            } else {

                String name = split[2];

                if (split.length > 3) {
                    for (int j = 3; j < split.length; j++) {
                        name += ":" + split[j];
                    }
                }

                // 去掉最后的空格
                name = name.substring(0, name.length() - 1);

                int lastFeatureId = featureIndex.get(split[1]);

                if ("/".equals(name)) {
                    name = "//";
                }

                if (name.contains("//")) {
                    name = name.replaceAll("//", "/XIEGANG/");
                }
                String featureName = toFeatureName(name.trim().split("/"), lastFeatureId);

                featureNames.add(Pair.with(split[1], featureName));

            }
        }

        return featureNames;

    }

    private String toFeatureName(String[] split, int lastFeatureId) throws Exception {

        StringBuilder result = new StringBuilder();

        for (String str : split) {
            if ("".equals(str)) {
                continue;
            } else if (str.length() == 1) {
                result.append(str.charAt(0));
            } else if (str.equals("XIEGANG")) {
                result.append('/');
            } else if (str.startsWith("num")) {
                result.append((char) (Config.NUM_BEGIN + ObjConver.getIntValue(str.replace("num", ""))));
            } else if (str.startsWith("en")) {
                result.append((char) (Config.EN_BEGIN + ObjConver.getIntValue(str.replace("en", ""))));
            } else if (str.startsWith("_x-")) {
                result.append(Config.BEGIN);
            } else if (str.startsWith("_x+")) {
                result.append(Config.END);
            } else {
                throw new Exception("can find feature named " + str + " in " + Arrays.toString(split));
            }
        }

        result.append((char) (lastFeatureId + Config.FEATURE_BEGIN));

        return result.toString();
    }

    /**
     * 加载特征标签转换
     * 
     * @param br
     * @return
     * @throws Exception
     */
    private int[] loadTagCoven(BufferedReader br) throws Exception {

        int[] conver = new int[Config.TAG_NUM + Config.TAG_NUM * Config.TAG_NUM];

        String temp = br.readLine();// #qrk#4

        // TODO: 这个是个写死的过程,如果标签发生改变需要重新来写这里
        for (int i = 0; i < Config.TAG_NUM; i++) {
            char c = br.readLine().split(":")[1].charAt(0);
            switch (c) {
                case 'S':
                    conver[i] = Config.S;
                    break;
                case 'B':
                    conver[i] = Config.B;
                    break;
                case 'M':
                    conver[i] = Config.M;
                    break;
                case 'E':
                    conver[i] = Config.E;
                    break;
                default:
                    throw new Exception("err tag named " + c + " in model " + temp);
            }
        }

        for (int i = Config.TAG_NUM; i < conver.length; i++) {
            conver[i] = conver[(i - 4) / Config.TAG_NUM] * Config.TAG_NUM + conver[i % Config.TAG_NUM] + Config.TAG_NUM;
        }

        return conver;
    }

    /**
     * 加载特征模板
     * 
     * @param br
     * @return
     * @throws IOException
     */
    private Map<String, Integer> loadConfig(BufferedReader br) throws IOException {

        Map<String, Integer> featureIndex = new HashMap<>();

        String temp = br.readLine();// #rdr#8/0/0

        int featureNum = ObjConver.getIntValue(StringUtil.matcherFirst("\\d+", temp)); // 找到特征个数

        List<int[]> list = new ArrayList<>();

        for (int i = 0; i < featureNum; i++) {
            temp = br.readLine();

            List<String> matcherAll = StringUtil.matcherAll("\\[.*?\\]", temp);

            if (matcherAll.isEmpty()) {
                continue;
            }

            int[] is = new int[matcherAll.size()];
            for (int j = 0; j < is.length; j++) {
                is[j] = ObjConver.getIntValue(StringUtil.matcherFirst("[-\\d]+", matcherAll.get(j)));
            }

            featureIndex.put(temp.split(":")[1], list.size());

            list.add(is);
        }

        int[][] template = new int[list.size()][0]; // 构建特征模板

        for (int i = 0; i < template.length; i++) {
            template[i] = list.get(i);
        }

        config = new Config(template);

        return featureIndex;
    }

    @Override
    public boolean checkModel(String modelPath) {

        try (InputStream is = IOUtil.getInputStream(modelPath)) {
            byte[] bytes = new byte[100];

            is.read(bytes);

            String string = new String(bytes);
            if (string.startsWith("#mdl#")) { // 加载crf++ 的txt类型的modle
                return true;
            }
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
        return false;
    }

}
