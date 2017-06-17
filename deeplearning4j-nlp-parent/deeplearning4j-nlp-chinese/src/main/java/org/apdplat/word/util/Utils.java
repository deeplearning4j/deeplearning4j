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

package org.apdplat.word.util;

import org.apdplat.word.recognition.StopWord;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.segmentation.Word;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.util.List;
import java.util.regex.Pattern;

/**
 * 工具类
 * @author 杨尚川
 */
public class Utils {
    private static final Logger LOGGER = LoggerFactory.getLogger(Utils.class);
    //至少出现一次中文字符，且以中文字符开头和结束
    private static final Pattern PATTERN_ONE = Pattern.compile("^[\\u4e00-\\u9fa5]+$");
    //至少出现两次中文字符，且以中文字符开头和结束
    private static final Pattern PATTERN_TWO = Pattern.compile("^[\\u4e00-\\u9fa5]{2,}$");
    /**
     * 至少出现一次中文字符，且以中文字符开头和结束
     * @param word
     * @return 
     */
    public static boolean isChineseCharAndLengthAtLeastOne(String word){
        if(PATTERN_ONE.matcher(word).find()){
            return true;
        }
        return false;
    }
    /**
     * 至少出现两次中文字符，且以中文字符开头和结束
     * @param word
     * @return 
     */
    public static boolean isChineseCharAndLengthAtLeastTwo(String word){
        if(PATTERN_TWO.matcher(word).find()){
            return true;
        }
        return false;
    }
    /**
     * 删除目录
     * @param dir 目录
     * @return 是否成功
     */
    public static boolean deleteDir(File dir) {
        if (dir.isDirectory()) {
            File[] children = dir.listFiles();
            for (File child : children) {
                boolean success = deleteDir(child);
                if (!success) {
                    return false;
                }
            }
        }
        return dir.delete();
    }
    /**
     *
     * 对文件进行分词
     * @param input 输入文件
     * @param output 输出文件
     * @param removeStopWords 是否移除停用词
     * @param segmentationAlgorithm 分词算法
     * @throws Exception
     */
    public static void seg(File input, File output, boolean removeStopWords, SegmentationAlgorithm segmentationAlgorithm) throws Exception{
        seg(input, output, removeStopWords, segmentationAlgorithm, null);
    }

    /**
     *
     * 对文件进行分词
     * @param input 输入文件
     * @param output 输出文件
     * @param removeStopWords 是否移除停用词
     * @param segmentationAlgorithm 分词算法
     * @param fileSegmentationCallback 分词结果回调
     * @throws Exception
     */
    public static void seg(File input, File output, boolean removeStopWords, SegmentationAlgorithm segmentationAlgorithm, FileSegmentationCallback fileSegmentationCallback) throws Exception{
        LOGGER.info("开始对文件进行分词："+input.toString());
        Segmentation segmentation = SegmentationFactory.getSegmentation(segmentationAlgorithm);
        float max=(float)Runtime.getRuntime().maxMemory()/1000000;
        float total=(float)Runtime.getRuntime().totalMemory()/1000000;
        float free=(float)Runtime.getRuntime().freeMemory()/1000000;
        String pre="执行之前剩余内存:"+max+"-"+total+"+"+free+"="+(max-total+free);
        //准备输出目录
        if(!output.getParentFile().exists()){
            output.getParentFile().mkdirs();
        }
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(input),"utf-8"));
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output),"utf-8"))){
            long size = Files.size(input.toPath());
            LOGGER.info("size:"+size);
            LOGGER.info("文件大小："+(float)size/1024/1024+" MB");
            int textLength=0;
            int progress=0;
            long start = System.currentTimeMillis();
            String line = null;
            while((line = reader.readLine()) != null){
                if("".equals(line.trim())){
                    writer.write("\n");
                    continue;
                }
                textLength += line.length();
                List<Word> words = segmentation.seg(line);
                if(removeStopWords){
                    //停用词过滤
                    StopWord.filterStopWords(words);
                }
                if(words == null){
                    continue;
                }
                for(Word word : words){
                    if(fileSegmentationCallback != null) {
                        fileSegmentationCallback.callback(word);
                    }
                    writer.write(word.getText()+" ");
                }
                writer.write("\n");
                progress += line.length();
                if( progress > 500000){
                    progress = 0;
                    LOGGER.info("分词进度："+(int)((float)textLength*2/size*100)+"%");
                }
            }
            long cost = System.currentTimeMillis() - start;
            float rate = textLength/cost;
            LOGGER.info("字符数目："+textLength);
            LOGGER.info("分词耗时："+getTimeDes(cost)+" 毫秒");
            LOGGER.info("分词速度："+rate+" 字符/毫秒");
        }
        max=(float)Runtime.getRuntime().maxMemory()/1000000;
        total=(float)Runtime.getRuntime().totalMemory()/1000000;
        free=(float)Runtime.getRuntime().freeMemory()/1000000;
        String post="执行之后剩余内存:"+max+"-"+total+"+"+free+"="+(max-total+free);
        LOGGER.info(pre);
        LOGGER.info(post);
        LOGGER.info("将文件 "+input.toString()+" 的分词结果保存到文件 "+output);
    }

    public static interface FileSegmentationCallback{
        public void callback(Word word);
    }

    /**
     * 根据毫秒数转换为自然语言表示的时间
     * @param ms 毫秒
     * @return 自然语言表示的时间
     */
    public static String getTimeDes(Long ms) {
        //处理参数为NULL的情况
        if(ms == null){
            return "";
        }
        int ss = 1000;
        int mi = ss * 60;
        int hh = mi * 60;
        int dd = hh * 24;

        long day = ms / dd;
        long hour = (ms - day * dd) / hh;
        long minute = (ms - day * dd - hour * hh) / mi;
        long second = (ms - day * dd - hour * hh - minute * mi) / ss;
        long milliSecond = ms - day * dd - hour * hh - minute * mi - second * ss;

        StringBuilder str=new StringBuilder();
        if(day>0){
            str.append(day).append("天,");
        }
        if(hour>0){
            str.append(hour).append("小时,");
        }
        if(minute>0){
            str.append(minute).append("分钟,");
        }
        if(second>0){
            str.append(second).append("秒,");
        }
        if(milliSecond>0){
            str.append(milliSecond).append("毫秒,");
        }
        if(str.length()>0){
            str=str.deleteCharAt(str.length()-1);
        }

        return str.toString();
    }
}
