package org.deeplearning4j.nn.layers.objdetect;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.Arrays;
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
        //Input activations shape, labels shape: [mb, depth, H, W]

        //Mask array must be present, with shape [minibatch,H,W]. Mask array is 1_i^B in YOLO paper - i.e., whether an object
        // is present (1) or not (0) in the specified grid cell (for specified example)
        if(maskArray == null){
            throw new IllegalStateException("No mask array is present: cannot compute score for YOLO network without a mask array");
        }

        if(maskArray.size(1) != input.size(2) || maskArray.size(2) != input.size(3)){
            throw new IllegalStateException("Mask array does not match input size: mask height/width (dimensions " +
                    "1/2 sizes) must match input array height/width (dimensions 2/3). Mask shape: "
                    + Arrays.toString(maskArray.shape()) + ", label shape: " + Arrays.toString(labels.shape()));
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

        Pair<INDArray,INDArray> label_tl_br = centerHwToTlbr(labelBoxYXCenterInImage, labelBoxHW);  //Shape: [mb, 2, H, W]

        //Determine top/left and bottom/right (x/y) of anchor bounding boxes, *in every grid cell*
        INDArray bbHW = layerConf().getBoundingBoxes();   //Bounding box priors: height + width. Shape: [B, 2]. *as a fraction of the total image dimensions*
        INDArray bbHWGrid = bbHW.broadcast(b, mb, 2, h, w);   //Shape: [B,2] -> [B,m,2,H,W]
        INDArray bbYXCenterImg = gridYX.broadcast(b, mb, 2, h, w); //Shape: [2,h,w] -> [B,mb,2,h,w]

        Pair<INDArray,INDArray> bb_tl_br = centerHwToTlbr(bbYXCenterImg, bbHWGrid);                 //Shape: [B, mb, 2, H, W]   - 2 dimension: Y,X


        //Calculate IOU (intersection over union - aka Jaccard index) - for the labels and bounding box priors
        //http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        INDArray iou = calculateIOU(label_tl_br.getFirst(), label_tl_br.getSecond(), bb_tl_br.getFirst(), bb_tl_br.getSecond(), labelBoxHW, bbHW, 2);
        //IOU shape: [minibatch, B, H, W]

        //Mask 1_ij^obj: isMax (dimension 1) + apply object present mask. Result: [minibatch, B, H, W]
        //In this mask: 1 if (a) object is present in cell [for each mb/H/W], and (b) for the anchor box with the max IOU
        INDArray mask1_ij_obj = Nd4j.getExecutioner().execAndReturn(new IsMax(iou, 1));
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(mask1_ij_obj, maskArray, mask1_ij_obj, 0,2,3));


        //

        return 0;
    }

    private static Pair<INDArray,INDArray> centerHwToTlbr(INDArray centerYX, INDArray hw){
        INDArray labelHalfHW = hw.div(2.0);
        INDArray topLeft = centerYX.sub(labelHalfHW);
        INDArray bottomRight = centerYX.add(labelHalfHW);
        return new Pair<>(topLeft, bottomRight);
    }

//    private static INDArray calculateIOU(INDArray centerYX1, INDArray hw1, INDArray centerYX2, INDArray hw2){
//    }

    private static INDArray calculateIOU(INDArray tl1, INDArray br1, INDArray tl2, INDArray br2, INDArray hw1, INDArray hw2, int yxDim){

        INDArray intersection = intersectionArea(tl1, br1, tl2, br2, yxDim);

        INDArray area1 = get(hw1, yxDim, 0).mul(get(hw1, yxDim, 1));
        INDArray area2 = get(hw2, yxDim, 0).mul(get(hw2, yxDim, 1));

        INDArray union = area1.add(area2).subi(intersection);

        INDArray iou = intersection.div(union);
        BooleanIndexing.replaceWhere(iou, 0.0, Conditions.isNan()); //Replace NaNs (0 intersection, 0 area etc) with 0s
        return iou;
    }

    private static INDArray intersectionArea(INDArray tl1, INDArray br1, INDArray tl2, INDArray br2, int yxDim){
        //Order: y, x
        int l = tl1.length();
        INDArray yxMax = Nd4j.getExecutioner().execAndReturn(new Max(tl1, tl2, Nd4j.createUninitialized(tl1.shape(), tl1.ordering()), l ));
        INDArray yxMin = Nd4j.getExecutioner().execAndReturn(new Min(br1, br2, Nd4j.createUninitialized(br1.shape(), br1.ordering()), l ));

        INDArray diffPlus1 = yxMin.sub(yxMax).addi(1.0);
        INDArray yTerm = get(diffPlus1, yxDim, 0);
        INDArray xTerm = get(diffPlus1, yxDim, 1);

        return xTerm.mul(yTerm);

    }

    private static INDArray get(INDArray in, int dim, int pos){
        INDArrayIndex[] indexes = new INDArrayIndex[in.rank()];
        for( int i=0; i<indexes.length; i++ ){
            if(i == dim){
                indexes[i] = NDArrayIndex.point(pos);
            } else {
                indexes[i] = NDArrayIndex.all();
            }
        }

        return in.get(indexes);
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
