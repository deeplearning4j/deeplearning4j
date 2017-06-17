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

/**
 * 中文分词算法
 * Chinese word segmentation algorithm
 * @author 杨尚川
 */
public enum SegmentationAlgorithm {
    /**
     * 正向最大匹配算法
     */
    MaximumMatching("正向最大匹配算法"),
    /**
     * 逆向最大匹配算法
     */
    ReverseMaximumMatching("逆向最大匹配算法"),
    /**
     * 正向最小匹配算法
     */
    MinimumMatching("正向最小匹配算法"),
    /**
     * 逆向最小匹配算法
     */
    ReverseMinimumMatching("逆向最小匹配算法"),
    /**
     * 双向最大匹配算法
     */
    BidirectionalMaximumMatching("双向最大匹配算法"),
    /**
     * 双向最小匹配算法
     */
    BidirectionalMinimumMatching("双向最小匹配算法"),
    /**
     * 双向最大最小匹配算法
     */
    BidirectionalMaximumMinimumMatching("双向最大最小匹配算法"),
    /**
     * 全切分算法
     */
    FullSegmentation("全切分算法"),

    /**
     * 最少词数算法
     */
    MinimalWordCount("最少词数算法"),

    /**
     * 最大Ngram分值算法
     */
    MaxNgramScore("最大Ngram分值算法"),

    /**
     * 针对纯英文文本的分词算法
     */
    PureEnglish("针对纯英文文本的分词算法");

    private SegmentationAlgorithm(String des){
        this.des = des;
    }
    private final String des;
    public String getDes() {
        return des;
    }
}
