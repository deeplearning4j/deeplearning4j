/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.standalone;

import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateExceptionHandler;
import freemarker.template.Version;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.api.Component;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.util.*;

/**
 * Idea: Render a set of components as a single static page.
 * The goal here is to provide a simple mechanism for exporting simple pages with static content (charts, etc),
 * where (a) the required UI components, and (b) the data itself, is embedded in the page
 * <p>
 * This is accomplished using a simple FreeMarker template
 *
 * @author Alex Black
 */
public class StaticPageUtil {

    private StaticPageUtil() {}

    /**
     * Given the specified components, render them to a stand-alone HTML page (which is returned as a String)
     *
     * @param components Components to render
     * @return Stand-alone HTML page, as a String
     */
    public static String renderHTML(Collection<Component> components) {
        return renderHTML(components.toArray(new Component[components.size()]));
    }

    /**
     * Given the specified components, render them to a stand-alone HTML page (which is returned as a String)
     *
     * @param components Components to render
     * @return Stand-alone HTML page, as a String
     */
    public static String renderHTML(Component... components) {
        try {
            return renderHTMLContent(components);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String renderHTMLContent(Component... components) throws Exception {

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        Configuration cfg = new Configuration(new Version(2, 3, 23));

        // Where do we load the templates from:
        cfg.setClassForTemplateLoading(StaticPageUtil.class, "");

        // Some other recommended settings:
        cfg.setIncompatibleImprovements(new Version(2, 3, 23));
        cfg.setDefaultEncoding("UTF-8");
        cfg.setLocale(Locale.US);
        cfg.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);

        ClassPathResource cpr = new ClassPathResource("assets/dl4j-ui.js");
        String scriptContents = IOUtils.toString(cpr.getInputStream(), "UTF-8");

        Map<String, Object> pageElements = new HashMap<>();
        List<ComponentObject> list = new ArrayList<>();
        int i = 0;
        for (Component c : components) {
            list.add(new ComponentObject(String.valueOf(i), mapper.writeValueAsString(c)));
            i++;
        }
        pageElements.put("components", list);
        pageElements.put("scriptcontent", scriptContents);


        Template template = cfg.getTemplate("staticpage.ftl");
        Writer stringWriter = new StringWriter();
        template.process(pageElements, stringWriter);

        return stringWriter.toString();
    }

    /**
     * A version of {@link #renderHTML(Component...)} that exports the resulting HTML to the specified path.
     *
     * @param outputPath Output path
     * @param components Components to render
     */
    public static void saveHTMLFile(String outputPath, Component... components) throws IOException {
        saveHTMLFile(new File(outputPath));
    }

    /**
     * A version of {@link #renderHTML(Component...)} that exports the resulting HTML to the specified File.
     *
     * @param outputFile Output path
     * @param components Components to render
     */
    public static void saveHTMLFile(File outputFile, Component... components) throws IOException {
        FileUtils.writeStringToFile(outputFile, renderHTML(components));
    }
}
