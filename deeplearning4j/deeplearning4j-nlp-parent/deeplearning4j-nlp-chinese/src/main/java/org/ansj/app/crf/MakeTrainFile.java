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

import org.ansj.app.crf.pojo.Element;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * 生成crf 或者是 wapiti的训练语聊工具.
 * 
 * 执行:java org.ansj.app.crf.MakeTrainFile [inputPath] [outputPath]
 * 
 * @author Ansj
 *
 */
public class MakeTrainFile {

    private static final Log logger = LogFactory.getLog();

    public static void main(String[] args) {

        String inputPath = "corpus.txt";

        String outputPath = "train.txt";

        if (args != null && args.length == 2) {
            inputPath = args[0];
            outputPath = args[1];
        }

        if (StringUtil.isBlank(inputPath) || StringUtil.isBlank(outputPath)) {
            logger.info("org.ansj.app.crf.MakeTrainFile [inputPath] [outputPath]");
            return;
        }
        try (BufferedReader reader = IOUtil.getReader(inputPath, "utf-8");
                        FileOutputStream fos = new FileOutputStream(outputPath)) {
            String temp = null;
            int i = 0;
            while ((temp = reader.readLine()) != null) {
                StringBuilder sb = new StringBuilder("\n");
                if (StringUtil.isBlank(temp)) {
                    continue;
                }
                if (i == 0) {
                    temp = StringUtil.trim(temp);
                }
                List<Element> list = Config.makeToElementList(temp, "\\s+");
                for (Element element : list) {
                    sb.append(element.nameStr() + " " + Config.getTagName(element.getTag()));
                    sb.append("\n");
                }
                fos.write(sb.toString().getBytes(IOUtil.UTF8));
                System.out.println(++i);
            }
        } catch (FileNotFoundException e) {
            logger.warn("文件没有找到", e);
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
    }

}
