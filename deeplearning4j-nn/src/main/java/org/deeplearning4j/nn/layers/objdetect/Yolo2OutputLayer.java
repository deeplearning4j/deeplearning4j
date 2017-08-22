package org.deeplearning4j.nn.layers.objdetect;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.List;

/**
 *
 * Label format: [minibatch, 5B+C, H, W]
 * Order for labels depth: [y,x,h,w,conf,(class labels)]
 * x = box center position (x axis), within the grid cell. (0,0) is top left of grid, (1,1) is bottom right of grid
 * y = as above, y axis
 * h = bounding box height, relative to whole image... in interval [0, 1]
 * w = as above, width
 *
 * @author Alex Black
 */
public class Yolo2OutputLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer> implements Serializable, IOutputLayer {

    //current input and label matrices
    @Setter @Getter
    protected INDArray labels;

    private double fullNetworkL1;
    private double fullNetworkL2;

    public Yolo2OutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return null;
    }

    @Override
    public INDArray activationMean() {
        return null;
    }

    @Override
    public INDArray activate(boolean training) {
        return null;
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training) {
        //Input activations shape: [mb, depth, H, W]

        //Mask array must be present, with shape [H,W]. Mask array is 1_i^B in YOLO paper - i.e., whether an object
        // is present (1) or not (0) in the specified grid cell
        if(maskArray == null){
            throw new IllegalStateException("No mask array is present: cannot compute score for YOLO network without a mask array");
        }

        if(maskArray.size(0) != input.size(2) || maskArray.size(1) != input.size(3)){
            throw new IllegalStateException("Mask array does not match input size: mask height/width (dimensions " +
                    "0/1 sizes) must match input array height/width (dimensions 2/3)");
        }

        int mb = input.size(0);
        int h = input.size(2);
        int w = input.size(3);
        int b = layerConf().getBoundingBoxes().size(0);

        double gridFracH = 1.0 / h;     //grid cell height, as a fraction of total image H
        double gridFracW = 1.0 / w;     //grid cell width, as a fraction of total image W

        //First: Infer Mask/indicator 1_{i,j}^obj
        //This is a [H,W,B] for each bonding box being responsible for the detection of an object
        //values are 1 if (a) an object is present in that cell, and (b) the specified bounding  box is
        // responsible

        INDArray xVector = Nd4j.linspace(0, 1.0-1.0/w, w);  //[0 to w-1]/w
        INDArray yVector = Nd4j.linspace(0, 1.0-1.0/h, h);  //[0 to h-1]/h
        INDArray gridYX = Nd4j.create(2,h,w);
        gridYX.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).putiColumnVector(yVector);
        gridYX.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).putiRowVector(xVector);



        //Determine top/left and bottom/right (x/y) of label bounding boxes - relative to the whole image
        //[0,0] is top left of whole image, [1,1] is bottom right of whole image
        //Label format: [minibatch, 5B+C, H, W]
        INDArray labelBoxYXCenterInGrid = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(0,1,true), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray labelBoxHW = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(2,3,true), NDArrayIndex.all(), NDArrayIndex.all());

        INDArray labelBoxYXCenterInImage = labelBoxYXCenterInGrid.dup();
        labelBoxYXCenterInImage.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
                .muli(gridFracH);
        labelBoxYXCenterInImage.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all())
                .muli(gridFracW);
        labelBoxYXCenterInImage.addi(gridYX);

//        INDArray labelHalfHW = labelBoxHW.div(2.0);
//        INDArray labelBoxTopLeftInImage = labelBoxYXCenterInImage.sub(labelHalfHW);
//        INDArray labelBoxBottomRightInImage = labelBoxYXCenterInImage.add(labelHalfHW);
        Pair<INDArray,INDArray> label_tl_br = centerHwToTlbr(labelBoxYXCenterInImage, labelBoxHW);  //Shape: [mb, 2, H, W]

        //Determine top/left and bottom/right (x/y) of anchor bounding boxes, *in every grid cell*
        INDArray bbHW = layerConf().getBoundingBoxes();   //Bounding box priors: height + width. Shape: [B, 2]. *as a fraction of the total image dimensions*
        INDArray bbHWGrid = bbHW.broadcast(b, mb, 2, h, w);   //Shape: [B,2] -> [B,m,2,H,W]
        INDArray bbYXCenterImg = gridYX.broadcast(b, mb, 2, h, w); //Shape: [2,h,w] -> [B,mb,2,h,w]

        Pair<INDArray,INDArray> bb_tl_br = centerHwToTlbr(bbYXCenterImg, bbHWGrid);                 //Shape: [B, mb, 2, H, W]
        





        INDArray bbX = null;    //TODO - infer X location of BB from H/W
        INDArray bbY = null;    //TODO - infer Y location of BB from H/W
        //Shape: [minibatch, 2, H, W]
            //Top left position of BB

        INDArray labelBoxXY2 = labelBoxXY.add(labelBoxHW);  //Bottom right position of BB
        INDArray labelBoxX = labels.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()); //Shape: [mb, H, W]
        INDArray labelBoxY = labels.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()); //Shape: [mb, H, W
        INDArray labelBoxH = labels.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()); //Shape: [mb, H, W]
        INDArray labelBoxW = labels.get(NDArrayIndex.all(), NDArrayIndex.point(3), NDArrayIndex.all(), NDArrayIndex.all()); //Shape: [mb, H, W

        //Calculate IOU (intersection over union - aka Jaccard index) - for the labels and bounding box priors
        //http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

        INDArray bbHBroadcast = bbHW.getColumn(0).broadcast(mb, b, h, w);
        INDArray bbWBroadcast = bbHW.getColumn(1).broadcast(mb, b, h, w);

        //Determine x/y coordinates of the intersection rectangle... neew broadcast min/max ops:
//        INDArray xA = Nd4j.getExecutioner().execAndReturn(new BroadcastMax)
        INDArray xMax = bbHBroadcast.dup();
        INDArray xMin = bbHBroadcast.dup();
        INDArray yMax = bbWBroadcast.dup();
        INDArray yMin = bbWBroadcast.dup();
        for( int i=0; i<b; i++ ){
            INDArray xMaxSub = xMax.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray xMinSub = xMin.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray yMaxSub = yMax.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray yMinSub = yMin.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
            Nd4j.getExecutioner().exec(new Max(xMaxSub, labelBoxX, xMaxSub, xMaxSub.length()));
            Nd4j.getExecutioner().exec(new Min(xMinSub, labelBoxX, xMinSub, xMinSub.length()));
            Nd4j.getExecutioner().exec(new Max(yMaxSub, labelBoxY, yMaxSub, yMaxSub.length()));
            Nd4j.getExecutioner().exec(new Min(yMinSub, labelBoxY, yMinSub, yMinSub.length()));
        }

        //At this point: (x/y)(Min/Max) are shape [mb, b, h, w]
        INDArray intersectionArea = xMin.sub(xMax).addi(1.0).muli(yMin.sub(yMax).addi(1.0));

        INDArray boxLabelArea = labelBoxH

        //Shape: [minibatch, B, H, W]
        INDArray mask1_ij_obj = null;




        return 0;
    }

    private static Pair<INDArray,INDArray> centerHwToTlbr(INDArray centerYX, INDArray hw){
        INDArray labelHalfHW = hw.div(2.0);
        INDArray topLeft = centerYX.sub(labelHalfHW);
        INDArray bottomRight = centerYX.add(labelHalfHW);
        return new Pair<>(topLeft, bottomRight);
    }

    @Override
    public void computeGradientAndScore(){


    }

    @Override
    public INDArray preOutput(boolean training) {
        return null;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2) {
        return null;
    }

    @Override
    public double f1Score(DataSet data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int numLabels() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(DataSetIterator iter) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int[] predict(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(DataSet data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, int[] labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}
