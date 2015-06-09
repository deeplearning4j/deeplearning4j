/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.plot;

import java.io.*;
import java.util.*;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


/**
 * Credit to :
 * http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
 *
 *
 * for visualizations
 * @author Adam Gibson
 *
 */
public class NeuralNetPlotter implements Serializable {

    private static 	ClassPathResource r = new ClassPathResource("/scripts/plot.py");
    private static final Logger log = LoggerFactory.getLogger(NeuralNetPlotter.class);
    private static   FilterRenderer render = new FilterRenderer();


    static {
        loadIntoTmp();
    }

    private static void loadIntoTmp() {

        File script = new File("/tmp/plot.py");

        try {
            List<String> lines = IOUtils.readLines(r.getInputStream());
            FileUtils.writeLines(script, lines);

        } catch (IOException e) {
            throw new IllegalStateException("Unable to load python file");

        }

    }

    protected String writeMatrix(INDArray matrix) throws IOException {
        String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
        File write = new File(filePath);
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
        write.deleteOnExit();
        for(int i = 0; i < matrix.rows(); i++) {
            INDArray row = matrix.getRow(i);
            StringBuilder sb = new StringBuilder();
            for(int j = 0; j < row.length(); j++) {
                sb.append(String.format("%.10f", row.getDouble(j)));
                if(j < row.length() - 1)
                    sb.append(",");
            }
            sb.append("\n");
            String line = sb.toString();
            bos.write(line.getBytes());
            bos.flush();
        }

        bos.close();
        return filePath;
    }

    public void renderFilter(INDArray w) {
        INDArray weightRender = w.dup();
        try {
            render.renderFilters(weightRender, "currimg.png", (int)Math.sqrt(weightRender.rows()) , (int) Math.sqrt( weightRender.columns()),10);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void renderGraph(String action, String data) throws Exception{
        String fileName = new ClassPathResource("python/plot.py").getFile().getAbsolutePath();

        try {
            log.info("Rendering Matrix histograms... ");
            Process is = Runtime.getRuntime().exec("python " + fileName + " " + action + " "+ data);
            Thread.sleep(10000);
            is.destroy();
            log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
            log.error("Std error " + IOUtils.readLines(is.getErrorStream()).toString());
        }catch(IOException e) {
            log.warn("Image closed");
            throw new RuntimeException(e);
        }

    }

    /**
     * Histograms of the given matrices. This is primarily used
     * for debugging gradients. You don't necessarily use this directly
     * @param titles the titles of the plots
     * @param matrices the matrices to plot
     */
    public void histogram(List<String> titles, INDArray[] matrices) throws Exception {
        String[] path = new String[matrices.length * 2];

        if(titles.size() != matrices.length)
            throw new IllegalArgumentException("Titles and matrix lengths must be equal");

        for(int i = 0; i < path.length - 1; i+=2) {
            path[i] = writeMatrix(matrices[i / 2].ravel());
            path[i + 1] = titles.get(i / 2);
        }
        String data = StringUtils.join(path,",");
        renderGraph("multi", data);

    }

    public void hist(Layer network,Gradient gradient) throws Exception{
        Set<String> vars = new TreeSet<>(gradient.gradientForVariable().keySet());
        List<String> titles = new ArrayList<>(vars);
        for(String s : vars) {
            titles.add(s + "-gradient");
        }
        histogram(
                titles,
                new INDArray[]{
                        network.getParam(DefaultParamInitializer.WEIGHT_KEY),
                        network.getParam(PretrainParamInitializer.BIAS_KEY),
                        network.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY),
                        gradient.gradientForVariable().get(DefaultParamInitializer.WEIGHT_KEY),
                        gradient.gradientForVariable().get(DefaultParamInitializer.BIAS_KEY),
                        gradient.gradientForVariable().get(PretrainParamInitializer.VISIBLE_BIAS_KEY)
                });
    }


    public void hist(Layer network) throws Exception {
        hist(network, network.gradient());
    }

    public void plotActivations(Layer network) throws Exception{

        if(network.input() == null)
            throw new IllegalStateException("Unable to plot; missing input");

        INDArray hbiasMean = network.activationMean();
        String data = writeMatrix(hbiasMean);

        renderGraph("hbias", data);

    }

    public void plotNetworkGradient(Layer network,Gradient gradient,int patchesPerRow) throws Exception {
        hist(network, gradient);
        plotActivations(network);

        FilterRenderer render = new FilterRenderer();
        try {
            INDArray w =  network.getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
            render.renderFilters(w, "currimg.png", (int) Math.sqrt(w.rows()),
                    (int) Math.sqrt(w.rows()), patchesPerRow);
        } catch (Exception e) {
            log.error("Unable to plot filter, continuing...",e);
        }
    }

    public void plotNetworkGradient(Layer network,INDArray gradient,int patchesPerRow) throws Exception{
        histogram(
                Arrays.asList("W", "w-gradient"),
                new INDArray[]{
                        network.getParam(DefaultParamInitializer.WEIGHT_KEY),
                        gradient
                });
        plotActivations(network);

        try {
            if(network.getParam(DefaultParamInitializer.WEIGHT_KEY).shape().length > 2) {
                INDArray w =  network.getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
                INDArray render2 = w.transpose();
                render.renderFilters(render2, "currimg.png", w.columns() , w.rows(),w.slices());

            }
            else
                render.renderFilters(network.getParam(DefaultParamInitializer.WEIGHT_KEY).dup(), "currimg.png", (int)Math.sqrt(network.getParam(DefaultParamInitializer.WEIGHT_KEY).rows()) , (int) Math.sqrt( network.getParam(DefaultParamInitializer.WEIGHT_KEY).rows()),patchesPerRow);


        } catch (Exception e) {
            log.error("Unable to plot filter, continuing...", e);
        }
    }


    /**
     * Scatter plot of the given matrices. This is primarily used
     * for debugging gradients. You don't necessarily use this directly
     * @param titles the titles of the plots
     * @param matrices the matrices to plot
     */
    public void scatter(String[] titles, INDArray[] matrices) {
        String[] path = new String[matrices.length * 2];

        try {
            if(titles.length != matrices.length)
                throw new IllegalArgumentException("Titles and matrix lengths must be equal");


            for(int i = 0; i < path.length - 1; i+=2) {
                path[i] = writeMatrix(matrices[i / 2].ravel());
                path[i + 1] = titles[i / 2];
            }
            String paths = StringUtils.join(path, ",");

            Process is = Runtime.getRuntime().exec("python /tmp/plot.py scatter " + paths);

            log.info("Rendering Matrix histograms... ");
            log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
            log.error(IOUtils.readLines(is.getErrorStream()).toString());


        }catch(IOException e) {
            throw new RuntimeException(e);
        }

    }



}
