package org.deeplearning4j;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.mkldnn.MKLDNNBatchNormHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.lang.reflect.Field;

import static junit.framework.TestCase.*;

public class TestBatchNormBp {

    @Test
    public void test(){
        Nd4j.getRandom().setSeed(12345);
//        INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, 4, 4);
        INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, 15, 15);
        INDArray mean = in.mean(0, 2, 3);   //Nd4j.rand(DataType.FLOAT, 3);
        INDArray var = in.var(0, 2, 3); //Nd4j.rand(DataType.FLOAT, 3);
        INDArray eps = Nd4j.rand(DataType.FLOAT, in.shape());
//        INDArray gamma = Nd4j.ones(DataType.FLOAT, 3);
//        INDArray beta = Nd4j.zeros(DataType.FLOAT, 3);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, 3);
        INDArray beta = Nd4j.rand(DataType.FLOAT, 3);
        double e = 1e-5;

        INDArray dLdIn = in.ulike();
        INDArray dLdm = mean.ulike();
        INDArray dLdv = var.ulike();
        INDArray dLdg = gamma.ulike();
        INDArray dLdb = beta.ulike();

        DynamicCustomOp op = DynamicCustomOp.builder("batchnorm_bp")
                .addInputs(in, mean, var, eps, gamma, beta)
                .addIntegerArguments(
                        1,          //Apply scale
                        1,           //Apply beta
                        1)           //Axis (NCHW)
                .addFloatingPointArguments(e)
                .addOutputs(dLdIn, dLdm, dLdv, dLdg, dLdb)
                .build();

        Nd4j.exec(op);
        System.out.println(dLdIn);
    }

    @Test
    public void compareImpls() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, 15, 15);
        INDArray mean = in.mean(0, 2, 3).reshape(1,3);
        INDArray var = in.var(0, 2, 3).reshape(1,3);
        INDArray eps = Nd4j.rand(DataType.FLOAT, in.shape());
        INDArray gamma = Nd4j.rand(DataType.FLOAT, 1,3);
        INDArray beta = Nd4j.rand(DataType.FLOAT, 1,3);
        double e = 1e-3;

        INDArray dLdIn = in.ulike();
        INDArray dLdm = mean.ulike();
        INDArray dLdv = var.ulike();
        INDArray dLdg = gamma.ulike();
        INDArray dLdb = beta.ulike();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new BatchNormalization.Builder().nIn(3).nOut(3).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        org.deeplearning4j.nn.layers.normalization.BatchNormalization bn = (org.deeplearning4j.nn.layers.normalization.BatchNormalization) net.getLayer(0);
        assertNotNull(bn.getHelper());
        Field f = bn.getClass().getDeclaredField("helper");
        f.setAccessible(true);
        f.set(bn, null);
        assertNull(bn.getHelper());


        MKLDNNBatchNormHelper h = new MKLDNNBatchNormHelper(DataType.FLOAT);

        net.output(in, true);
        bn.setInput(in, LayerWorkspaceMgr.noWorkspaces());
        Pair<Gradient,INDArray> p = net.backpropGradient(eps, LayerWorkspaceMgr.noWorkspaces());

        h.preOutput(in, true, new long[]{1,3}, gamma, beta, mean, var, 0.5, e, LayerWorkspaceMgr.noWorkspaces());
        Pair<Gradient,INDArray> pmkl = h.backpropGradient(in, eps, new long[]{1,3}, gamma, beta, dLdg, dLdb, e, LayerWorkspaceMgr.noWorkspaces());

        INDArray dldin_dl4j = p.getSecond();

        System.out.println("dl4j == mkldnn: " + p.getSecond().equals(pmkl.getSecond()));
    }

}
