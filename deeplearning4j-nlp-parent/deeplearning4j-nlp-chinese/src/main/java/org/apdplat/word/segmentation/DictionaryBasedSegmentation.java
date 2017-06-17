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

import org.apdplat.word.dictionary.Dictionary;

/**
 * 基于词典的中文分词接口
 * Dictionary Based Chinese Word Segmentation Interface
 * @author 杨尚川
 */
public interface DictionaryBasedSegmentation extends Segmentation{
    /**
     * 为基于词典的中文分词接口指定词典操作接口
     * @param dictionary 词典操作接口
     */
    public void setDictionary(Dictionary dictionary);

    /**
     * 获取词典操作接口
     * @return 词典操作接口
     */
    public Dictionary getDictionary();
}
