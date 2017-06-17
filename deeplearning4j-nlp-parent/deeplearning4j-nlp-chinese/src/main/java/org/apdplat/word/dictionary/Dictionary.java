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

package org.apdplat.word.dictionary;

import java.util.List;

/**
 * 词典操作接口
 * @author 杨尚川
 */
public interface Dictionary {
    /**
     * 词典中的词的最大长度，即有多少个字符
     * @return 长度
     */
    public int getMaxLength();

    /**
     * 判断指定的文本是不是一个词
     * @param item 文本
     * @param start 指定的文本从哪个下标索引开始
     * @param length 指定的文本的长度
     * 比如：contains("我爱写程序",  3, 2);
     * 表示的意思是“程序”是不是一个定义在词典中的词
     * @return 是否
     */
    public boolean contains(String item, int start, int length);

    /**
     * 判断文本是不是一个词
     * @param item 文本
     * @return 是否
     */
    public boolean contains(String item);

    /**
     * 批量将词加入词典
     * @param items 集合中的每一个元素是一个词
     */
    public void addAll(List<String> items);

    /**
     * 将单个词加入词典
     * @param item 词
     */
    public void add(String item);

    /**
     * 批量将词从词典中删除
     * @param items 集合中的每一个元素是一个词
     */
    public void removeAll(List<String> items);

    /**
     * 将单个词从词典中删除
     * @param item 词
     */
    public void remove(String item);

    /**
     * 清空词典中的所有的词
     */
    public void clear();
}
