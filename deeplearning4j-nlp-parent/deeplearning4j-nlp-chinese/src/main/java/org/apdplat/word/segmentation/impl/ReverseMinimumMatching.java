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
 * 基于词典的逆向最小匹配算法
 * Dictionary-based reverse minimum matching algorithm
 * @author 杨尚川
 */
public class ReverseMinimumMatching extends AbstractSegmentation{

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.ReverseMinimumMatching;
    }
    @Override
    public List<Word> segImpl(String text) {
        Stack<Word> result = new Stack<>();
        //文本长度
        final int textLen=text.length();
        //从未分词的文本中截取的长度
        int len=1;
        //剩下未分词的文本的索引
        int start=textLen-len;
        //只要有词未切分完就一直继续
        while(start>=0){
            //用长为len的字符串查词典
            while(!getDictionary().contains(text, start, len) && !RecognitionTool.recog(text, start, len)){
                //如果查不到，则长度加一后继续
                //索引向前移动一个字，然后继续
                len++;
                start--;
                //如果长度为词典最大长度且在词典中未找到匹配
                //或已经遍历完剩下的文本且在词典中未找到匹配
                //则按长度为一切分
                if(len>getInterceptLength() || start<0){
                    //重置截取长度为一
                    //向后移动start索引
                    start+=len-1;
                    len=1;
                    break;
                }
            }
            addWord(result, text, start, len);
            //每一次成功切词后都要重置开始索引位置
            start--;
            //每一次成功切词后都要重置截取长度
            len=1;
        }
        len=result.size();
        List<Word> list = new ArrayList<>(len);
        for(int i=0;i<len;i++){
            list.add(result.pop());
        }
        return list;        
    }
    public static void main(String[] args){
        String text = "他不管三七二十一就骂她是二百五，我就无语了，真是个二货。他还问我：“杨老师，‘二货’是什么意思？”";
        if(args !=null && args.length == 1){
            text = args[0];
        }
        ReverseMinimumMatching m = new ReverseMinimumMatching();
        System.out.println(m.seg(text).toString());
    }
}
