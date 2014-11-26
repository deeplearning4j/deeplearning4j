package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
public class MultiLayerConfiguration implements Serializable {

    private int[] hiddenLayerSizes;
    private List<NeuralNetConfiguration> confs;

    private MultiLayerConfiguration() {}



    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }

    public int[] getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    public void setHiddenLayerSizes(int[] hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public List<NeuralNetConfiguration> getConfs() {
        return confs;
    }

    public void setConfs(List<NeuralNetConfiguration> confs) {
        this.confs = confs;
    }


    public String toJson() {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static MultiLayerConfiguration fromJson(String json) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public static class Builder {

        private List<NeuralNetConfiguration> confs;
        private int[] hiddenLayerSizes;

        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;

        }


        public Builder hiddenLayerSizes(int[] hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            return this;
        }

        public MultiLayerConfiguration build() {
            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            conf.hiddenLayerSizes = this.hiddenLayerSizes;
            return conf;

        }


    }


}
