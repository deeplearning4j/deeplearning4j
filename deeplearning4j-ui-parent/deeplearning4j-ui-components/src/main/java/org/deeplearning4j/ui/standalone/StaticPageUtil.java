package org.deeplearning4j.ui.standalone;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.api.Component;

import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateExceptionHandler;
import freemarker.template.Version;

import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.util.*;

/**
 * Idea: Render a set of components as a single static page.
 * The goal here is to provide a simple mechanism for exporting simple pages with static content (charts, etc),
 * where (a) the required UI components, and (b) the data itself, is embedded in the page
 *
 * This is accomplished using a simple FreeMarker template
 */
public class StaticPageUtil {

    public static String renderHTML(Component... components) throws Exception {

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
        String scriptContents = IOUtils.toString(cpr.getInputStream(),"UTF-8");

        Map<String, Object> pageElements = new HashMap<>();
        List<ComponentObject> list = new ArrayList<>();
        int i=0;
        for(Component c : components){
            list.add(new ComponentObject(String.valueOf(i), mapper.writeValueAsString(c)));
            i++;
        }
        pageElements.put("components",list);
        pageElements.put("scriptcontent",scriptContents);


        Template template = cfg.getTemplate("staticpage.ftl");
        Writer stringWriter = new StringWriter();
        template.process(pageElements,stringWriter);

        return stringWriter.toString();
    }

    public static void saveHTMLFile(String outputPath, Component... components) throws Exception {
        FileUtils.writeStringToFile(new File(outputPath), renderHTML(components));
    }
}
