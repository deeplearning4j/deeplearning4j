package org.deeplearning4j.nn.layers.objdetect;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 *
 * Label format: [minibatch, 4+C, H, W]
 * Order for labels depth: [x1,y1,x2,y2,(class labels)]
 * x1 = box top left position
 * y1 = as above, y axis
 * x2 = box bottom right position
 * y2 = as above y axis
 * Note: labels are represented as fraction of total image - (0,0) is top left, (1,1) is bottom right
 *
 * Input format: [minibatch, 5B+C, H, W]    ->      Reshape to [minibatch, B, 5+C, H, W]
 * Layout for dimension 2 (of size 5+C) after reshaping: [xInGrid,yInGrid,w,h]
 *
 *
 * Masks: not required. Infer presence or absence from labels.
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
        //Labels shape: [mb, 4B+C, H, W],

        //Infer mask array from labels. Mask array is 1_i^B in YOLO paper - i.e., whether an object is present in that
        // grid location or not. Here: we are using the fact that class labels are one-hot, and assume that values are
        // all 0s if no class label is present
        int size1 = labels.size(1);
        INDArray classLabels = labels.get(all(), interval(4,size1), all(), all());
        INDArray maskArray = classLabels.sum(1); //Shape: [minibatch, H, W]

        int mb = input.size(0);
        int h = input.size(2);
        int w = input.size(3);
        int b = layerConf().getBoundingBoxes().size(0);
        int c = labels.size(1)-4*b;


        // ----- Step 1: Labels format conversion -----
        //First: Convert labels/ground truth (x1,y1,x2,y2) from "fraction of total image" format to center format, as
        // fraction of total image
        //0.5 * ([x1,y1]+[x2,y2])   ->      shape: [mb, 2, H, W]
        INDArray labelTLXYImg = labels.get(all(),interval(0,1,true), all(), all());
        INDArray labelBRXYImg = labels.get(all(),interval(2,3,true), all(), all());
        INDArray labelCenterXYImg = labelTLXYImg.add(labelBRXYImg).muli(0.5);

        //Then convert label centers from "fraction of total image" to "fraction of grid", which are used in position loss
        INDArray wh = Nd4j.create(new double[]{w,h});
        INDArray labelsCenterXYInGrid = Nd4j.getExecutioner().execAndReturn( new BroadcastMulOp(labelCenterXYImg, wh,
                        Nd4j.createUninitialized(labelCenterXYImg.shape(), labelCenterXYImg.ordering()), 1 ));
        labelsCenterXYInGrid.subi(Transforms.floor(labelsCenterXYInGrid,true));

        //Also infer size/scale (label w/h) from (x1,y1,x2,y2) format to (w,h) format
        // Then apply sqrt ready for use in loss function
        INDArray labelWHSqrt = labels.get(all(),interval(2,3,true), all(), all()).sub(
                    labels.get(all(),interval(0,1,true), all(), all()));
        Transforms.sqrt(labelWHSqrt, false);



        // ----- Step 2: apply activation functions to network output activations -----
        //Reshape from [minibatch, 5B+C, H, W] to [minibatch, B, 5+C, H, W]
        INDArray input5 = input.dup('c').reshape(mb, b, 5+c, h, w);

        // Sigmoid for x/y centers
        INDArray predictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        Transforms.sigmoid(predictedXYCenterGrid, false);

        //Exponential for w/h (for: boxPrior * exp(input))
        INDArray predictedWH = input5.get(all(), all(), interval(2,4), all(), all());
        Transforms.exp(predictedWH, false);

        //Calculate predicted top/left and bottom/right in overall image
            //First: calculate top/left  value for each grid location. gridXY contains
        INDArray xVector = Nd4j.linspace(0, 1.0-1.0/w, w);  //[0 to w-1]/w
        INDArray yVector = Nd4j.linspace(0, 1.0-1.0/h, h);  //[0 to h-1]/h
        INDArray gridYX = Nd4j.create(2,h,w);
        gridYX.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).putiRowVector(xVector);
        gridYX.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).putiColumnVector(yVector);




        INDArray predictedXYCenterImage = Broadcast.div(predictedXYCenterGrid, wh,
                Nd4j.createUninitialized(predictedXYCenterGrid.shape(), predictedXYCenterGrid.ordering()), 1 ));
        Broadcast.add(predictedXYCenterImage, gridYX, predictedXYCenterImage, 2,3,4); // [2,H,W] to [minibatch, B, 2, H, W]

        INDArray halfWidth = predictedWH.mul(0.5);
        INDArray predictedTLXYImage = predictedXYCenterImage.sub(halfWidth);
        INDArray predictedBRXYImage = halfWidth.addi(predictedXYCenterImage);

        //Apply sqrt to W/H in preparation for loss function
        INDArray predictedWHSqrt = Transforms.sqrt(predictedWH, false);



        // ----- Step 3: Calculate IOU(predicted, labels) to infer 1_ij^obj mask array (for loss function) -----

//        INDArray predictedTL =

        //Calculate IOU (intersection over union - aka Jaccard index) - for the labels and bounding box priors
        //http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        INDArray iou = calculateIOU5d(labelTLXYImg, labelBRXYImg, predictedTLXYImage, predictedBRXYImage);
        //IOU shape: [minibatch, B, H, W]

        //Mask 1_ij^obj: isMax (dimension 1) + apply object present mask. Result: [minibatch, B, H, W]
        //In this mask: 1 if (a) object is present in cell [for each mb/H/W], and (b) for the anchor box with the max IOU
        INDArray mask1_ij_obj = Nd4j.getExecutioner().execAndReturn(new IsMax(iou, 1));
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(mask1_ij_obj, maskArray, mask1_ij_obj, 0,2,3));


        //Calculate predicted box locations + dimensions from activations
        //Input shape: [minibatch, 5B+C, H, W]
        //Layout for depth dimension: [5Y, 5X, 5H, 5W, 5C]

        //Pull out both Y and X components
        INDArray predictedYXCenterInGridCell = input.get(all(), interval(0,2*b,true), all(), all());
        predictedYXCenterInGridCell = Transforms.sigmoid(predictedYXCenterInGridCell, true);    //Shape: [minibatch, 2, h, w]

        INDArray predictedYXCenter2d = predictedYXCenterInGridCell.dup('c').reshape('c', mb, 2*b*h*w);
        //Create broadcasted version of labels:
        INDArray broadcastLabelsYX = Nd4j.createUninitialized(predictedYXCenterInGridCell.shape(), 'c');
        INDArray bLabelsY = broadcastLabelsYX.get(all(), interval(0,4), all(), all());  //Shape: [minibatch, 4, H, W]
        Nd4j.getExecutioner().execAndReturn(new BroadcastCopyOp(bLabelsY, labelBoxYCenterInGridCell, bLabelsY, 0,2,3));  //
        INDArray bLabelsX = broadcastLabelsYX.get(all(), interval(4,8), all(), all());  //Shape: [minibatch, 4, H, W]
        Nd4j.getExecutioner().execAndReturn(new BroadcastCopyOp(bLabelsX, labelBoxXCenterInGridCell, bLabelsX, 0,2,3));  //
        INDArray labelYXCenter2d = broadcastLabelsYX.dup('c').reshape('c', mb, 2*b*h*w);

        //And 2d version of the mask1_ij_obj: [minibatch,B,H,W] -> ???



        //Calculate the loss:
        double positionLoss = layerConf().getLossPositionScale().computeScore(labelYXCenter2d, predictedYXCenter2d, null, null, false );
        double sizeScaleLoss = 0.0;
        double confidenceLoss = 0.0;
        double classPredictionLoss = 0.0;

        double loss = layerConf().getLambdaCoord() * (positionLoss + sizeScaleLoss)
                + confidenceLoss
                + classPredictionLoss;

        return 0;
    }

    private static INDArray calculateIOU5d(INDArray tl1, INDArray br1, INDArray tl2, INDArray br2){

        INDArray intersection = intersectionArea5d(tl1, br1, tl2, br2);

        INDArray area1 = tl1.
        INDArray area1 = get(hw1, yxDim, 0).mul(get(hw1, yxDim, 1));
        INDArray area2 = get(hw2, yxDim, 0).mul(get(hw2, yxDim, 1));

        INDArray union = area1.add(area2).subi(intersection);

        INDArray iou = intersection.div(union);
        BooleanIndexing.replaceWhere(iou, 0.0, Conditions.isNan()); //Replace NaNs (0 intersection, 0 area etc) with 0s
        return iou;
    }

    private static INDArray intersectionArea5d(INDArray tl1, INDArray br1, INDArray tl2, INDArray br2){
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
                indexes[i] = point(pos);
            } else {
                indexes[i] = all();
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
