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

import java.util.List;

/**
 * 中文分词接口
 * Chinese Word Segmentation Interface
 * @author 杨尚川
 */
public interface Segmentation {
    /**
     * 将文本切分为词
     * @param text 文本
     * @return 词
     */
    public List<Word> seg(String text);
    /**
     * 分词器使用的算法
     * @return 分词算法
     */
    public SegmentationAlgorithm getSegmentationAlgorithm();
}
