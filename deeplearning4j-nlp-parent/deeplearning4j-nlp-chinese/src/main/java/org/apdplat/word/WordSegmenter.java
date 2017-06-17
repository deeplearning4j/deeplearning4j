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

package org.apdplat.word;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.recognition.StopWord;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 中文分词基础入口
 * 默认使用双向最大匹配算法
 * 也可指定其他分词算法
 * @author 杨尚川
 */
public class WordSegmenter {
    private static final Logger LOGGER = LoggerFactory.getLogger(WordSegmenter.class);    
    /**
     * 对文本进行分词，保留停用词
     * 可指定其他分词算法
     * @param text 文本
     * @param segmentationAlgorithm 分词算法
     * @return 分词结果
     */
    public static List<Word> segWithStopWords(String text, SegmentationAlgorithm segmentationAlgorithm){
        return SegmentationFactory.getSegmentation(segmentationAlgorithm).seg(text);
    }
    /**
     * 对文本进行分词，保留停用词
     * 使用双向最大匹配算法
     * @param text 文本
     * @return 分词结果
     */
    public static List<Word> segWithStopWords(String text){
        return SegmentationFactory.getSegmentation(SegmentationAlgorithm.MaxNgramScore).seg(text);
    }
    /**
     * 对文本进行分词，移除停用词
     * 可指定其他分词算法
     * @param text 文本
     * @param segmentationAlgorithm 分词算法
     * @return 分词结果
     */
    public static List<Word> seg(String text, SegmentationAlgorithm segmentationAlgorithm){        
        List<Word> words = SegmentationFactory.getSegmentation(segmentationAlgorithm).seg(text);
        //停用词过滤
        StopWord.filterStopWords(words);
        return words;
    }
    /**
     * 对文本进行分词，移除停用词
     * 使用双向最大匹配算法
     * @param text 文本
     * @return 分词结果
     */
    public static List<Word> seg(String text){
        List<Word> words = SegmentationFactory.getSegmentation(SegmentationAlgorithm.MaxNgramScore).seg(text);
        //停用词过滤
        StopWord.filterStopWords(words);
        return words;
    }
    /**
     * 对文件进行分词，保留停用词
     * 可指定其他分词算法
     * @param input 输入文件
     * @param output 输出文件
     * @param segmentationAlgorithm 分词算法
     * @throws Exception 
     */
    public static void segWithStopWords(File input, File output, SegmentationAlgorithm segmentationAlgorithm) throws Exception{
        Utils.seg(input, output, false, segmentationAlgorithm);
    }
    /**
     * 对文件进行分词，保留停用词
     * 使用双向最大匹配算法
     * @param input 输入文件
     * @param output 输出文件
     * @throws Exception 
     */
    public static void segWithStopWords(File input, File output) throws Exception{
        Utils.seg(input, output, false, SegmentationAlgorithm.MaxNgramScore);
    }
    /**
     * 对文件进行分词，移除停用词
     * 可指定其他分词算法
     * @param input 输入文件
     * @param output 输出文件
     * @param segmentationAlgorithm 分词算法
     * @throws Exception 
     */
    public static void seg(File input, File output, SegmentationAlgorithm segmentationAlgorithm) throws Exception{
        Utils.seg(input, output, true, segmentationAlgorithm);
    }
    /**
     * 对文件进行分词，移除停用词
     * 使用双向最大匹配算法
     * @param input 输入文件
     * @param output 输出文件
     * @throws Exception 
     */
    public static void seg(File input, File output) throws Exception{
        Utils.seg(input, output, true, SegmentationAlgorithm.MaxNgramScore);
    }
    private static void demo(){
        long start = System.currentTimeMillis();
        List<String> sentences = new ArrayList<>();
        sentences.add("杨尚川是APDPlat应用级产品开发平台的作者");
        sentences.add("他说的确实在理");
        sentences.add("提高人民生活水平");
        sentences.add("他俩儿谈恋爱是从头年元月开始的");
        sentences.add("王府饭店的设施和服务是一流的");
        sentences.add("和服务于三日后裁制完毕，并呈送将军府中");
        sentences.add("研究生命的起源");
        sentences.add("他明天起身去北京");
        sentences.add("在这些企业中国有企业有十个");
        sentences.add("他站起身来");
        sentences.add("他们是来查金泰撞人那件事的");
        sentences.add("行侠仗义的查金泰远近闻名");
        sentences.add("长春市长春节致辞");
        sentences.add("他从马上摔下来了,你马上下来一下");
        sentences.add("乒乓球拍卖完了");
        sentences.add("咬死猎人的狗");
        sentences.add("地面积了厚厚的雪");
        sentences.add("这几块地面积还真不小");
        sentences.add("大学生活象白纸");
        sentences.add("结合成分子式");
        sentences.add("有意见分歧");
        sentences.add("发展中国家兔的计划");
        sentences.add("明天他将来北京");
        sentences.add("税收制度将来会更完善");
        sentences.add("依靠群众才能做好工作");
        sentences.add("现在是施展才能的好机会");
        sentences.add("把手举起来");
        sentences.add("请把手拿开");
        sentences.add("这个门把手坏了");
        sentences.add("茶杯的把手断了");
        sentences.add("将军任命了一名中将");
        sentences.add("产量三年中将增长两倍");
        sentences.add("以新的姿态出现在世界东方");
        sentences.add("使节约粮食进一步形成风气");
        sentences.add("反映了一个人的精神面貌");
        sentences.add("美国加州大学的科学家发现");
        sentences.add("我好不挺好");
        sentences.add("木有"); 
        sentences.add("下雨天留客天天留我不留");
        sentences.add("叔叔亲了我妈妈也亲了我");
        sentences.add("白马非马");
        sentences.add("学生会写文章");
        sentences.add("张掖市民陈军");
        sentences.add("张掖市明乐县");
        sentences.add("中华人民共和国万岁万岁万万岁");
        sentences.add("word是一个中文分词项目，作者是杨尚川，杨尚川的英文名叫ysc");
        sentences.add("江阴毛纺厂成立了保持党员先进性爱国主义学习小组,在江阴道路管理局协助下,通过宝鸡巴士公司,与蒙牛酸酸乳房山分销点组成了开放性交互式的讨论组, 认为google退出中国事件赤裸裸体现了帝国主义的文化侵略,掀起了爱国主义的群众性高潮。");
        sentences.add("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作");
        sentences.add("商品和服务");
        sentences.add("结婚的和尚未结婚的");
        sentences.add("买水果然后来世博园");
        sentences.add("中国的首都是北京");
        sentences.add("老师说明天下午休息");
        sentences.add("今天下雨");
        int i=1;
        for(String sentence : sentences){
            List<Word> words = segWithStopWords(sentence);
            LOGGER.info((i++)+"、切分句子: "+sentence);
            LOGGER.info("    切分结果："+words);
        }
        long cost = System.currentTimeMillis() - start;
        LOGGER.info("耗时: "+cost+" 毫秒");
    }
    public static void processCommand(String... args) {
        if(args == null || args.length < 1){
            LOGGER.info("命令不正确");
            return;
        }
        try{
            switch(args[0].trim().charAt(0)){
                case 'd':
                    demo();
                    break;
                case 't':
                    if(args.length < 2){
                        showUsage();
                    }else{
                        StringBuilder str = new StringBuilder();
                        for(int i=1; i<args.length; i++){
                            str.append(args[i]).append(" ");
                        }
                        List<Word> words = segWithStopWords(str.toString());
                        LOGGER.info("切分句子："+str.toString());
                        LOGGER.info("切分结果："+words.toString());
                    }
                    break;
                case 'f':
                    if(args.length != 3){
                        showUsage();
                    }else{
                        segWithStopWords(new File(args[1]), new File(args[2]));
                    }
                    break;
                default:
                    StringBuilder str = new StringBuilder();
                    for(String a : args){
                        str.append(a).append(" ");
                    }
                    List<Word> words = segWithStopWords(str.toString());
                    LOGGER.info("切分句子："+str.toString());
                    LOGGER.info("切分结果："+words.toString());
                    break;
            }
        }catch(Exception e){
            showUsage();
        }
    }
    private static void run(String encoding) {
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, encoding))){
            String line = null;
            while((line = reader.readLine()) != null){
                if("exit".equals(line)){
                    System.exit(0);
                    LOGGER.info("退出");
                    return;
                }
                if(line.trim().equals("")){
                    continue;
                }
                processCommand(line.split(" "));
                showUsage();
            }
        } catch (IOException ex) {
            LOGGER.error("程序中断：", ex);
        }
    }
    private static void showUsage(){
        LOGGER.info("");
        LOGGER.info("********************************************");
        LOGGER.info("用法: command [text] [input] [output]");
        LOGGER.info("命令command的可选值为：demo、text、file");
        LOGGER.info("命令可使用缩写d t f，如不指定命令，则默认为text命令，对输入的文本分词");
        LOGGER.info("demo");
        LOGGER.info("text 杨尚川是APDPlat应用级产品开发平台的作者");
        LOGGER.info("file d:/text.txt d:/word.txt");
        LOGGER.info("exit");
        LOGGER.info("********************************************");
        LOGGER.info("输入命令后回车确认：");
    }
    public static void main(String[] args) {
        String encoding = "utf-8";
        if(args==null || args.length == 0){
            showUsage();
            run(encoding);
        }else if(Charset.isSupported(args[0])){
            showUsage();
            run(args[0]);
        }else{
            processCommand(args);
            //非交互模式，退出JVM
            System.exit(0);
        }
    }
}