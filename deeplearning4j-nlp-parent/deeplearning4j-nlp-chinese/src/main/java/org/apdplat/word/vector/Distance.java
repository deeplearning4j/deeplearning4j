/**
 * 
 * APDPlat - Application Product Development Platform
 * Copyright (c) 2013, 杨尚川, yang-shangchuan@qq.com
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

package org.apdplat.word.vector;

import org.apdplat.word.analysis.*;
import org.apdplat.word.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * 计算词和词的相似度
 * @author 杨尚川
 */
public class Distance {
    private static final Logger LOGGER = LoggerFactory.getLogger(Distance.class);
    private TextSimilarity textSimilarity = null;
    private Map<String, String> model = null;
    private int limit = 15;
    public Distance(TextSimilarity textSimilarity, String model) throws Exception {
        this.textSimilarity = textSimilarity;
        this.model = parseModel(model);
    }

    public void setTextSimilarity(TextSimilarity textSimilarity) {
        LOGGER.info("设置相似度算法为："+textSimilarity.getClass().getName());
        this.textSimilarity = textSimilarity;
    }

    public void setLimit(int limit) {
        LOGGER.info("设置显示结果条数为："+limit);
        this.limit = limit;
    }

    private Map<String, String> parseModel(String model) throws Exception {
        Map<String, String> map = new HashMap<>();
        LOGGER.info("开始初始化模型");
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(model), "utf-8"))) {
            String line = null;
            while ((line = reader.readLine()) != null) {
                String[] attr = line.split(" : ");
                if (attr == null || attr.length != 2) {
                    LOGGER.error("错误数据：" + line);
                    continue;
                }
                String key = attr[0];
                String value = attr[1];
                value = value.substring(1, value.length() - 1);
                map.put(key, value);
            }
        }
        LOGGER.info("模型初始化完成");
        return map;
    }
    private void tip(){
        LOGGER.info("可通过输入命令sa=cos来指定相似度算法，可用的算法有：");
        LOGGER.info("   1、sa=cos，余弦相似度");
        LOGGER.info("   2、sa=edi，编辑距离");
        LOGGER.info("   3、sa=euc，欧几里得距离");
        LOGGER.info("   4、sa=sim，简单共有词");
        LOGGER.info("   5、sa=jac，Jaccard相似性系数");
        LOGGER.info("   6、sa=man，曼哈顿距离");
        LOGGER.info("   7、sa=shh，SimHash + 汉明距离");
        LOGGER.info("   8、sa=ja，Jaro距离");
        LOGGER.info("   9、sa=jaw，Jaro–Winkler距离");
        LOGGER.info("   10、sa=sd，Sørensen–Dice系数");
        LOGGER.info("可通过输入命令limit=15来指定显示结果条数");
        LOGGER.info("可通过输入命令exit退出程序");
        LOGGER.info("输入要查询的词或命令：");
    }
    private void interact(String encoding) throws Exception{
        tip();
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, encoding))){
            String line = null;
            while((line = reader.readLine()) != null){
                if("exit".equals(line)){
                    System.exit(0);
                }
                if(line.startsWith("limit=")){
                    try{
                        setLimit(Integer.parseInt(line.replace("limit=", "").trim()));
                    }catch (Exception e){
                        LOGGER.error("指令不正确，数字非法");
                    }
                    continue;
                }
                if(line.startsWith("sa=")){
                    switch (line.substring(3)){
                        case "cos": setTextSimilarity(new CosineTextSimilarity());continue;
                        case "edi": setTextSimilarity(new EditDistanceTextSimilarity());continue;
                        case "euc": setTextSimilarity(new EuclideanDistanceTextSimilarity());continue;
                        case "sim": setTextSimilarity(new SimpleTextSimilarity());continue;
                        case "jac": setTextSimilarity(new JaccardTextSimilarity());continue;
                        case "man": setTextSimilarity(new ManhattanDistanceTextSimilarity());continue;
                        case "shh": setTextSimilarity(new SimHashPlusHammingDistanceTextSimilarity());continue;
                        case "ja": setTextSimilarity(new JaroDistanceTextSimilarity());continue;
                        case "jaw": setTextSimilarity(new JaroWinklerDistanceTextSimilarity());continue;
                        case "sd": setTextSimilarity(new SørensenDiceCoefficientTextSimilarity());continue;
                    }
                    continue;
                }
                String value = model.get(line);
                if(value == null){
                    LOGGER.info("没有对应的词："+line);
                }else{
                    LOGGER.info("计算词向量：" + value);
                    LOGGER.info("显示结果数目：" + limit);
                    LOGGER.info(line+" 的相关词（"+textSimilarity.getClass().getSimpleName()+"）：");
                    LOGGER.info("----------------------------------------------------------");
                    long start = System.currentTimeMillis();
                    List<String> list = compute(value, limit);
                    long cost = System.currentTimeMillis() - start;
                    AtomicInteger i = new AtomicInteger();
                    for(String element : list){
                        LOGGER.info("\t"+i.incrementAndGet()+"、"+element);
                    }
                    LOGGER.info("----------------------------------------------------------");
                    LOGGER.info("耗时：" + Utils.getTimeDes(cost));
                    LOGGER.info("----------------------------------------------------------");
                }
                tip();
            }
        }
    }
    public List<String> compute(String words, int limit){
        Map<String, Float> wordVec = new HashMap<>();
        String[] ws = words.split(", ");
        for(String w : ws){
            String[] attr = w.split(" ");
            String k = attr[0];
            float v = Float.parseFloat(attr[1]);
            wordVec.put(k, v);
        }       
        Map<String, Double> result = new HashMap<>();
        for(String key : model.keySet()){
            //词向量
            String value = model.get(key);
            String[] elements = value.split(", ");
            Map<String, Float> vec = new HashMap<>();
            for(String element : elements){
                String[] attr = element.split(" ");
                String k = attr[0];
                float v = Float.parseFloat(attr[1]);
                vec.put(k, v);
            }
            //忽略维度小于10的词向量
            if(vec.size()<10){
                continue;
            }
            //计算距离，也就是相似度分值
            double score = textSimilarity.similarScore(wordVec, vec);
            if(score > 0){
                result.put(key, score);
            }
        }
        if(result.isEmpty()){
            LOGGER.info("没有相似词");
            return Collections.emptyList();
        }
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("相似词数：" + result.size());
        }
        //按分值排序
        List<Entry<String, Double>> list = result.entrySet().parallelStream().sorted((a,b)->b.getValue().compareTo(a.getValue())).collect(Collectors.toList());
        //限制结果数目
        if(limit > list.size()){
            limit = list.size();
        }
        //转换为字符串
        List<String> retValue = new ArrayList<>(limit);
        for(int i=0; i< limit; i++){
            retValue.add(list.get(i).getKey()+" "+list.get(i).getValue());
        }
        return retValue;
    }
    public static void main(String[] args) throws Exception{
        String model = "data/vector.txt";
        String encoding = "gbk";
        if(args.length == 1){
            model = args[0];
        }
        if(args.length == 2){
            model = args[0];
            encoding = args[1];
        }
        Distance distance = new Distance(new EditDistanceTextSimilarity(), model);
        distance.interact(encoding);
    }
}
