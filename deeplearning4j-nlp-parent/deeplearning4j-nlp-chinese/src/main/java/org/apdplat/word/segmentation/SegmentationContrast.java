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

package org.apdplat.word.segmentation;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;

/**
 * 对比各种分词算法的分词结果
 * @author 杨尚川
 */
public class SegmentationContrast {
    public static Map<String, String> seg(String text) {
        Map<String, String> results = new HashMap<>();
        for(SegmentationAlgorithm segmentationAlgorithm : SegmentationAlgorithm.values()){
            String result = SegmentationFactory.getSegmentation(segmentationAlgorithm).seg(text).toString();
            results.put(segmentationAlgorithm.getDes(), result);
        }
        return results;
    }
    public static void dump(Map<String, String> map){
        System.out.println("***************************************************");
        System.out.println("切分效果对比：");
        System.out.println("***************************************************");
        map.keySet().stream().sorted().forEach(sa -> System.out.println(sa + " : " + map.get(sa)));
        System.out.println("***************************************************");
    }
    public static void run(String encoding) {
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, encoding))){
            String line = null;
            while((line = reader.readLine()) != null){
                if("exit".equals(line)){
                    System.exit(0);
                    return;
                }
                if(line.trim().equals("")){
                    continue;
                }
                dump(seg(line));
                showUsage();
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    public static void showUsage(){
        System.out.println("输入exit退出程序");
        System.out.println("输入要分词的文本后回车确认：");
    }
    public static void main(String[] args) {
        dump(seg("独立自主和平等互利的原则"));
        String encoding = "utf-8";
        if(args==null || args.length == 0){
            showUsage();
            run(encoding);
        }else if(Charset.isSupported(args[0])){
            showUsage();
            run(args[0]);
        }
    }
}
