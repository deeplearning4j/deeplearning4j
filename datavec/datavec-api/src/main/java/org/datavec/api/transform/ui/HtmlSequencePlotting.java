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

package org.datavec.api.transform.ui;

import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateExceptionHandler;
import freemarker.template.Version;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.components.RenderableComponentLineChart;
import org.datavec.api.transform.ui.components.RenderableComponentTable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.StringWriter;
import java.io.Writer;
import java.util.*;

/**
 * A simple utility for plotting DataVec sequence data to HTML files.
 * Each file contains only one sequence. Each column is plotted separately; only numerical and categorical columns are
 * plotted.
 *
 * @author Alex Black
 */
public class HtmlSequencePlotting {

    private HtmlSequencePlotting() {

    }

    /**
     * Create a HTML file with plots for the given sequence.
     *
     * @param title    Title of the page
     * @param schema   Schema for the data
     * @param sequence Sequence to plot
     * @return HTML file as a string
     */
    public static String createHtmlSequencePlots(String title, Schema schema, List<List<Writable>> sequence)
                    throws Exception {
        Configuration cfg = new Configuration(new Version(2, 3, 23));

        // Where do we load the templates from:
        cfg.setClassForTemplateLoading(HtmlSequencePlotting.class, "/templates/");

        // Some other recommended settings:
        cfg.setIncompatibleImprovements(new Version(2, 3, 23));
        cfg.setDefaultEncoding("UTF-8");
        cfg.setLocale(Locale.US);
        cfg.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);


        Map<String, Object> input = new HashMap<>();
        input.put("pagetitle", title);

        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);

        List<DivObject> divs = new ArrayList<>();
        List<String> divNames = new ArrayList<>();

        //First: create table for schema
        int n = schema.numColumns();
        String[][] table = new String[n / 2 + n % 2][6]; //Number, name, type; 2 columns

        List<ColumnMetaData> meta = schema.getColumnMetaData();
        for (int i = 0; i < meta.size(); i++) {
            int o = i % 2;
            table[i / 2][o * 3] = String.valueOf(i);
            table[i / 2][o * 3 + 1] = meta.get(i).getName();
            table[i / 2][o * 3 + 2] = meta.get(i).getColumnType().toString();
        }

        for (int i = 0; i < table.length; i++) {
            for (int j = 0; j < table[i].length; j++) {
                if (table[i][j] == null) {
                    table[i][j] = "";
                }
            }
        }


        RenderableComponentTable rct = new RenderableComponentTable.Builder().table(table)
                        .header("#", "Name", "Type", "#", "Name", "Type").backgroundColor("#FFFFFF")
                        .headerColor("#CCCCCC").colWidthsPercent(8, 30, 12, 8, 30, 12).border(1).padLeftPx(4)
                        .padRightPx(4).build();
        divs.add(new DivObject("tablesource", ret.writeValueAsString(rct)));

        //Create the plots
        double[] x = new double[sequence.size()];
        for (int i = 0; i < x.length; i++) {
            x[i] = i;
        }

        for (int i = 0; i < n; i++) {
            double[] lineData;
            switch (meta.get(i).getColumnType()) {
                case Integer:
                case Long:
                case Double:
                case Float:
                case Time:
                    lineData = new double[sequence.size()];
                    for (int j = 0; j < lineData.length; j++) {
                        lineData[j] = sequence.get(j).get(i).toDouble();
                    }
                    break;
                case Categorical:
                    //This is a quick-and-dirty way to plot categorical variables as a line chart
                    List<String> stateNames = ((CategoricalMetaData) meta.get(i)).getStateNames();
                    lineData = new double[sequence.size()];
                    for (int j = 0; j < lineData.length; j++) {
                        String state = sequence.get(j).get(i).toString();
                        int idx = stateNames.indexOf(state);
                        lineData[j] = idx;
                    }
                    break;
                case Bytes:
                case String:
                case Boolean:
                default:
                    //Skip
                    continue;
            }

            String name = meta.get(i).getName();

            String chartTitle = "Column: \"" + name + "\" - Column Type: " + meta.get(i).getColumnType();
            if (meta.get(i).getColumnType() == ColumnType.Categorical) {
                List<String> stateNames = ((CategoricalMetaData) meta.get(i)).getStateNames();
                StringBuilder sb = new StringBuilder(chartTitle);
                sb.append(" - (");
                for (int j = 0; j < stateNames.size(); j++) {
                    if (j > 0) {
                        sb.append(", ");
                    }
                    sb.append(j).append("=").append(stateNames.get(j));
                }
                sb.append(")");
                chartTitle = sb.toString();
            }

            RenderableComponentLineChart lc = new RenderableComponentLineChart.Builder().title(chartTitle)
                            .addSeries(name, x, lineData).build();

            String divname = "plot_" + i;

            divs.add(new DivObject(divname, ret.writeValueAsString(lc)));
            divNames.add(divname);
        }

        input.put("divs", divs);
        input.put("divnames", divNames);

        //Current date/time, UTC
        DateTimeFormatter formatter = DateTimeFormat.forPattern("YYYY-MM-dd HH:mm:ss zzz").withZone(DateTimeZone.UTC);
        long currTime = System.currentTimeMillis();
        String dateTime = formatter.print(currTime);
        input.put("datetime", dateTime);

        Template template = cfg.getTemplate("sequenceplot.ftl");

        //Process template to String
        Writer stringWriter = new StringWriter();
        template.process(input, stringWriter);

        return stringWriter.toString();
    }

    /**
     * Create a HTML file with plots for the given sequence and write it to a file.
     *
     * @param title    Title of the page
     * @param schema   Schema for the data
     * @param sequence Sequence to plot
     */
    public static void createHtmlSequencePlotFile(String title, Schema schema, List<List<Writable>> sequence,
                    File output) throws Exception {
        String s = createHtmlSequencePlots(title, schema, sequence);
        FileUtils.writeStringToFile(output, s);
    }
}
