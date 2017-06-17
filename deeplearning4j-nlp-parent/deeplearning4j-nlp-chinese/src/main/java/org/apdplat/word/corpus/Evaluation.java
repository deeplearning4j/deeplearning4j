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

package org.apdplat.word.corpus;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apdplat.word.WordSegmenter;
import org.apdplat.word.recognition.Punctuation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.slf4j.LoggerFactory;

/**
 * 利用人工标注的语料库
 * 对分词算法效果进行评估
 * 评估采用的测试文本有253 3709行，共2837 4490个字符
 * 评估结果位于target/evaluation目录下：
 * corpus-text.txt为分好词的人工标注文本，词之间以空格分隔
 * test-text.txt为测试文本，是把corpus-text.txt以标点符号分隔为多行的结果
 * standard-text.txt为测试文本对应的人工标注文本，作为分词是否正确的标准
 * result-text-***，***为各种分词算法名称，这是word分词结果
 * perfect-result-***，***为各种分词算法名称，这是分词结果和人工标注标准完全一致的文本
 * wrong-result-***，***为各种分词算法名称，这是分词结果和人工标注标准不一致的文本	
 * @author 杨尚川
 */
public class Evaluation {
    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger(Evaluation.class);

    public static void main(String[] args) throws Exception{
        //分好词的人工标注文本，词之间以空格分隔
        String corpusText = "target/evaluation/corpus-text.txt";
        //测试文本，是把corpus-text.txt以标点符号分隔为多行的结果
        String testText = "target/evaluation/test-text.txt";
        //测试文本对应的人工标注文本，作为分词是否正确的标准
        String standardText = "target/evaluation/standard-text.txt";
        //word分词结果
        String resultText = "target/evaluation/result-text-";
        //分词结果和人工标注标准完全一致的文本
        String perfectResult = "target/evaluation/perfect-result-";
        //分词结果和人工标注标准不一致的文本
        String wrongResult = "target/evaluation/wrong-result-";
        //评估结果位于target/evaluation目录下：
        Path path = Paths.get("target/evaluation");
        if(!Files.exists(path)){
            Files.createDirectory(path);
        }
        //1、抽取文本
        ExtractText.extractFromCorpus(corpusText, " ", false);
        //2、生成测试数据集和标准数据集
        int textCharCount = generateDataset(corpusText, testText, standardText);
        List<EvaluationResult> result = new ArrayList<>();
        for(SegmentationAlgorithm segmentationAlgorithm : SegmentationAlgorithm.values()){
            long start = System.currentTimeMillis();
            //3、对测试数据集进行分词
            WordSegmenter.segWithStopWords(new File(testText), new File(resultText+segmentationAlgorithm.name()+".txt"), segmentationAlgorithm);
            long cost = System.currentTimeMillis() - start;
            float rate = textCharCount/(float)cost;
            //4、分词效果评估
            EvaluationResult evaluationResult = evaluation(resultText+segmentationAlgorithm.name()+".txt", standardText, perfectResult+segmentationAlgorithm.name()+".txt", wrongResult+segmentationAlgorithm.name()+".txt");
            evaluationResult.setSegmentationAlgorithm(segmentationAlgorithm);
            evaluationResult.setSegSpeed(rate);
            result.add(evaluationResult);
        }
        //5、输出测试报告
        LOGGER.info("*************************************************************************************************************");
        Collections.sort(result);
        for(int i=0; i<result.size(); i++){
            LOGGER.info(result.get(i).toString());
            if(i < result.size()-1){
                LOGGER.info("");
            }
        }
        LOGGER.info("*************************************************************************************************************");
    }
    /**
     * 生成测试数据集和标准数据集
     * @param file 已分词文本，词之间空格分隔
     * @param test 生成测试数据集文件路径
     * @param standard 生成标准数据集文件路径
     * @return 测试数据集字符数
     */
    public static int generateDataset(String file, String test, String standard){
        int textCharCount=0;
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file),"utf-8"));
            BufferedWriter testWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(test),"utf-8"));
            BufferedWriter standardWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(standard),"utf-8"))){
            String line;
            int duplicateCount=0;
            Set<String> set = new HashSet<>();
            while( (line = reader.readLine()) != null ){
                //不把空格当做标点符号
                List<String> list = Punctuation.seg(line, false, ' ');
                for(String item : list){
                    item = item.trim();
                    //忽略空行和长度为一的行
                    if("".equals(item)
                            || item.length()==1){
                        continue;
                    }
                    //忽略重复的内容
                    if(set.contains(item)){
                        duplicateCount++;
                        continue;
                    }
                    set.add(item);
                    String testItem = item.replaceAll(" ", "");
                    textCharCount += testItem.length();
                    testWriter.write(testItem+"\n");
                    standardWriter.write(item+"\n");
                }
            }
            LOGGER.info("重复行数为："+duplicateCount);
        } catch (IOException ex) {
            LOGGER.error("生成测试数据集和标准数据集失败：", ex);
        }
        return textCharCount;
    }
    /**
     * 分词效果评估
     * @param resultText 实际分词结果文件路径
     * @param standardText 标准分词结果文件路径
     * @param perfectResult 分词完美内容保存文件路径
     * @param wrongResult 分词错误内容保存文件路径
     * @return 评估结果
     */
    public static EvaluationResult evaluation(String resultText, String standardText, String perfectResult, String wrongResult) {
        long start = System.currentTimeMillis();
        int perfectLineCount=0;
        int wrongLineCount=0;
        int perfectCharCount=0;
        int wrongCharCount=0;
        try(BufferedReader resultReader = new BufferedReader(new InputStreamReader(new FileInputStream(resultText),"utf-8"));
            BufferedReader standardReader = new BufferedReader(new InputStreamReader(new FileInputStream(standardText),"utf-8"));
            BufferedWriter perfectResultWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(perfectResult),"utf-8"));
            BufferedWriter wrongResultWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(wrongResult),"utf-8"))){
            String result;
            while( (result = resultReader.readLine()) != null ){
                result = result.trim();
                String standard = standardReader.readLine().trim();
                if(result.equals("")){
                    continue;
                }
                if(result.equals(standard)){
                    //分词结果和标准一模一样
                    perfectResultWriter.write(standard+"\n");
                    perfectLineCount++;
                    perfectCharCount+=standard.replaceAll("\\s+", "").length();
                }else{
                    //分词结果和标准不一样
                    wrongResultWriter.write("实际分词结果："+result+"\n");
                    wrongResultWriter.write("标准分词结果："+standard+"\n");
                    wrongLineCount++;
                    wrongCharCount+=standard.replaceAll("\\s+", "").length();
                }
            }
        } catch (IOException ex) {
            LOGGER.error("分词效果评估失败：", ex);
        }
        long cost = System.currentTimeMillis() - start;
        int totalLineCount = perfectLineCount+wrongLineCount;
        int totalCharCount = perfectCharCount+wrongCharCount;
        LOGGER.info("评估耗时："+cost+" 毫秒");
        EvaluationResult er = new EvaluationResult();
        er.setPerfectCharCount(perfectCharCount);
        er.setPerfectLineCount(perfectLineCount);
        er.setTotalCharCount(totalCharCount);
        er.setTotalLineCount(totalLineCount);
        er.setWrongCharCount(wrongCharCount);
        er.setWrongLineCount(wrongLineCount);     
        return er;
    }
    /**
     * 分词效果评估
     * @param resultText 实际分词结果文件路径
     * @param standardText 标准分词结果文件路径
     * @return 评估结果
     */
    public static EvaluationResult evaluation(String resultText, String standardText) {
        int perfectLineCount=0;
        int wrongLineCount=0;
        int perfectCharCount=0;
        int wrongCharCount=0;
        try(BufferedReader resultReader = new BufferedReader(new InputStreamReader(new FileInputStream(resultText),"utf-8"));
            BufferedReader standardReader = new BufferedReader(new InputStreamReader(new FileInputStream(standardText),"utf-8"))){
            String result;
            while( (result = resultReader.readLine()) != null ){
                result = result.trim();
                String standard = standardReader.readLine().trim();
                if(result.equals("")){
                    continue;
                }
                if(result.equals(standard)){
                    //分词结果和标准一模一样
                    perfectLineCount++;
                    perfectCharCount+=standard.replaceAll("\\s+", "").length();
                }else{
                    //分词结果和标准不一样
                    wrongLineCount++;
                    wrongCharCount+=standard.replaceAll("\\s+", "").length();
                }
            }
        } catch (IOException ex) {
            LOGGER.error("分词效果评估失败：", ex);
        }
        int totalLineCount = perfectLineCount+wrongLineCount;
        int totalCharCount = perfectCharCount+wrongCharCount;
        EvaluationResult er = new EvaluationResult();
        er.setPerfectCharCount(perfectCharCount);
        er.setPerfectLineCount(perfectLineCount);
        er.setTotalCharCount(totalCharCount);
        er.setTotalLineCount(totalLineCount);
        er.setWrongCharCount(wrongCharCount);
        er.setWrongLineCount(wrongLineCount);     
        return er;
    }
}