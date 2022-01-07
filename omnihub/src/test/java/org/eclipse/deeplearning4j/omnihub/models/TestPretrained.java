package org.eclipse.deeplearning4j.omnihub.models;

import org.junit.jupiter.api.Test;

public class TestPretrained {


    @Test
    public void testPretrained() throws Exception {
        //Pretrained.samediff().gpt2(true);
        Pretrained.dl4j().vgg19noTop(true);
    }

}
