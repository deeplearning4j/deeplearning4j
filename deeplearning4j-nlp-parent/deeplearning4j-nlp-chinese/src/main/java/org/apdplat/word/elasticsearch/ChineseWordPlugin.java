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

package org.apdplat.word.elasticsearch;

import org.elasticsearch.common.component.LifecycleComponent;
import org.elasticsearch.common.inject.Module;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.analysis.AnalysisModule;
import org.elasticsearch.plugins.Plugin;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/**
 * 中文分词组件（word）的ElasticSearch插件
 * @author 杨尚川
 */
public class ChineseWordPlugin extends Plugin {
    private final Settings settings;
    public ChineseWordPlugin(Settings settings) {
        this.settings = settings;
    }
    @Override
    public String name() {
        return "word";
    }
    @Override
    public String description() {
        return "中文分词组件（word）";
    }
    @Override
    public Collection<Module> nodeModules() {
        return Collections.<Module>singletonList(new ChineseWordIndicesAnalysisModule());
    }
    @Override
    public Collection<Class<? extends LifecycleComponent>> nodeServices() {
        Collection<Class<? extends LifecycleComponent>> services = new ArrayList<>();
        return services;
    }
    @Override
    public Collection<Module> indexModules(Settings indexSettings) {
        return Collections.<Module>singletonList(new ChineseWordIndicesAnalysisModule());
    }
    public void onModule(AnalysisModule module) {
        module.addProcessor(new ChineseWordAnalysisBinderProcessor());
    }
}