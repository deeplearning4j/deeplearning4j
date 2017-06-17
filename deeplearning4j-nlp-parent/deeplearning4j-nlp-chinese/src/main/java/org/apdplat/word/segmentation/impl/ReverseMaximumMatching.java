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

package org.apdplat.word.segmentation.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import org.apdplat.word.recognition.RecognitionTool;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.Word;

/**
 * 基于词典的逆向最大匹配算法
 * Dictionary-based reverse maximum matching algorithm
 * @author 杨尚川
 */
public class ReverseMaximumMatching extends AbstractSegmentation{

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.ReverseMaximumMatching;
    }
    @Override
    public List<Word> segImpl(String text) {
        Stack<Word> result = new Stack<>();
        //文本长度
        final int textLen=text.length();
        //从未分词的文本中截取的长度
        int len=getInterceptLength();
        //剩下未分词的文本的索引
        int start=textLen-len;
        //处理文本长度小于最大词长的情况
        if(start<0){
            start=0;
        }
        if(len>textLen-start){
            //如果未分词的文本的长度小于截取的长度
            //则缩短截取的长度
            len=textLen-start;
        }
        //只要有词未切分完就一直继续
        while(start>=0 && len>0){
            //用长为len的字符串查词典，并做特殊情况识别
            while(!getDictionary().contains(text, start, len) && !RecognitionTool.recog(text, start, len)){
                //如果长度为一且在词典中未找到匹配
                //则按长度为一切分
                if(len==1){
                    break;
                }
                //如果查不到，则长度减一
                //索引向后移动一个字，然后继续
                len--;
                start++;
            }
            addWord(result, text, start, len);
            //每一次成功切词后都要重置截取长度
            len=getInterceptLength();            
            if(len>start){
                //如果未分词的文本的长度小于截取的长度
                //则缩短截取的长度
                len=start;
            }
            //每一次成功切词后都要重置开始索引位置
            //从待分词文本中向前移动最大词长个索引
            //将未分词的文本纳入下次分词的范围
            start-=len;
        }
        len=result.size();
        List<Word> list = new ArrayList<>(len);
        for(int i=0;i<len;i++){
            list.add(result.pop());
        }
        return list;        
    }        
    public static void main(String[] args){
        String text = "'软件工程四大圣经'：《设计模式》、《反模式》、《重构》、《解析极限编程》。其中《设计模式》和《重构》号称'软工双雄'。";
        if(args !=null && args.length == 1){
            text = args[0];
        }
        ReverseMaximumMatching m = new ReverseMaximumMatching();
        System.out.println(m.seg(text).toString());
    }
}
