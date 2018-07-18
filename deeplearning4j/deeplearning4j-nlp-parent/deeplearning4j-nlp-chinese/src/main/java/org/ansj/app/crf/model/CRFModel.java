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

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;

/**
 * 加载ansj格式的crfmodel,目前此model格式是通过crf++ 或者wapiti生成的
 * 
 * @author Ansj
 *
 */
public class CRFModel extends Model {

    public static final String VERSION = "ansj1";

    @Override
    public CRFModel loadModel(String modelPath) throws Exception {
        try (InputStream is = IOUtil.getInputStream(modelPath)) {
            loadModel(is);
            return this;
        }
    }

    @Override
    public CRFModel loadModel(InputStream is) throws Exception {
        long start = System.currentTimeMillis();
        try (ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(is))) {
            ois.readUTF();
            this.status = (float[][]) ois.readObject();
            int[][] template = (int[][]) ois.readObject();
            this.config = new Config(template);
            int win = 0;
            int size = 0;
            String name = null;
            featureTree = new SmartForest<float[]>();
            float[] value = null;
            do {
                win = ois.readInt();
                size = ois.readInt();
                for (int i = 0; i < size; i++) {
                    name = ois.readUTF();
                    value = new float[win];
                    for (int j = 0; j < value.length; j++) {
                        value[j] = ois.readFloat();
                    }
                    featureTree.add(name, value);
                }
            } while (win == 0 || size == 0);
            logger.info("load crf model ok ! use time :" + (System.currentTimeMillis() - start));
        }
        return this;
    }

    @Override
    public boolean checkModel(String modelPath) {
        try (FileInputStream fis = new FileInputStream(modelPath)) {
            ObjectInputStream inputStream = new ObjectInputStream(new GZIPInputStream(fis));
            String version = inputStream.readUTF();
            if (version.equals("ansj1")) { // 加载ansj,model
                return true;
            }
        } catch (ZipException ze) {
            logger.warn("解压异常", ze);
        } catch (FileNotFoundException e) {
            logger.warn("文件没有找到", e);
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
        return false;
    }

}
