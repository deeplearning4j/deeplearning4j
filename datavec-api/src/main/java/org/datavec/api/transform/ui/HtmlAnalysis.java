/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.ui;

import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateExceptionHandler;
import freemarker.template.Version;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.schema.Schema;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.ui.components.RenderableComponentHistogram;
import org.datavec.api.transform.ui.components.RenderableComponentTable;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.*;
import java.util.*;

/**
 * Created by Alex on 25/03/2016.
 */
public class HtmlAnalysis {

    public static String createHtmlAnalysisString(DataAnalysis analysis) throws Exception {
        Configuration cfg = new Configuration(new Version(2, 3, 23));

        // Where do we load the templates from:
        cfg.setClassForTemplateLoading(HtmlAnalysis.class, "/templates/");

        // Some other recommended settings:
        cfg.setIncompatibleImprovements(new Version(2, 3, 23));
        cfg.setDefaultEncoding("UTF-8");
        cfg.setLocale(Locale.US);
        cfg.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);


        Map<String, Object> input = new HashMap<>();

        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);

        List<ColumnAnalysis> caList = analysis.getColumnAnalysis();
        Schema schema = analysis.getSchema();

        int n = caList.size();
        String[][] table = new String[n][3];

        List<DivObject> divs = new ArrayList<>();
        List<String> histogramDivNames = new ArrayList<>();

        for( int i=0; i<n; i++ ){
            ColumnAnalysis ca = caList.get(i);
            String name = schema.getName(i);    //namesList.get(i);
            ColumnType type = schema.getType(i);

            table[i][0] = name;
            table[i][1] = type.toString();
            table[i][2] = ca.toString().replaceAll(",", ", ");  //Hacky work-around to improve display in HTML table

            double[] buckets;
            long[] counts;

            switch(type){
                case String:
                    StringAnalysis sa = (StringAnalysis)ca;
                    buckets = sa.getHistogramBuckets();
                    counts = sa.getHistogramBucketCounts();
                    break;
                case Integer:
                    IntegerAnalysis ia = (IntegerAnalysis)ca;
                    buckets = ia.getHistogramBuckets();
                    counts = ia.getHistogramBucketCounts();
                    break;
                case Long:
                    LongAnalysis la = (LongAnalysis) ca;
                    buckets = la.getHistogramBuckets();
                    counts = la.getHistogramBucketCounts();
                    break;
                case Double:
                    DoubleAnalysis da = (DoubleAnalysis)ca;
                    buckets = da.getHistogramBuckets();
                    counts = da.getHistogramBucketCounts();
                    break;
                case Categorical:
                case Time:
                case Bytes:
                    buckets = null;
                    counts = null;
                    break;
                default:
                    throw new RuntimeException("Invalid/unknown column type: " + type);
            }

            if(buckets != null){
                RenderableComponentHistogram.Builder histBuilder = new RenderableComponentHistogram.Builder();

                for( int j=0; j<counts.length; j++ ){
                    histBuilder.addBin(buckets[j],buckets[j+1],counts[j]);
                }

                histBuilder.margins(60,60,90,20);

                RenderableComponentHistogram hist = histBuilder.title(name).build();

                String divName = "histdiv_" + name.replaceAll("\\W","");
                divs.add(new DivObject(divName, ret.writeValueAsString(hist)));
                histogramDivNames.add(divName);
            }
        }

        //Create the summary table
        RenderableComponentTable rct = new RenderableComponentTable.Builder()
                .table(table)
                .header("Column Name", "Column Type", "Column Analysis")
                .backgroundColor("#FFFFFF")
                .headerColor("#CCCCCC")
                .colWidthsPercent(20,10,70)
                .border(1)
                .padLeftPx(4).padRightPx(4)
                .build();

        divs.add(new DivObject("tablesource",ret.writeValueAsString(rct)));

        input.put("divs", divs);
        input.put("histogramIDs", histogramDivNames);

        //Current date/time, UTC
        DateTimeFormatter formatter = DateTimeFormat.forPattern("YYYY-MM-dd HH:mm:ss zzz").withZone(DateTimeZone.UTC);
        long currTime = System.currentTimeMillis();
        String dateTime = formatter.print(currTime);
        input.put("datetime",dateTime);

        Template template = cfg.getTemplate("analysis.ftl");

        //Process template to String
        Writer stringWriter = new StringWriter();
        template.process(input, stringWriter);

        return stringWriter.toString();
    }

    public static void createHtmlAnalysisFile(DataAnalysis dataAnalysis, File output ) throws Exception {

        String str = createHtmlAnalysisString(dataAnalysis);

        FileUtils.writeStringToFile(output, str);
    }

}
