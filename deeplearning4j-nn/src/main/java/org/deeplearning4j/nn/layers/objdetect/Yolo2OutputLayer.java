package org.deeplearning4j.nn.layers.objdetect;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
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
 * Note: labels are represented as a multiple of grid size - for a 13x13 grid, (0,0) is top left, (13,13) is bottom right
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

    private static final Gradient EMPTY_GRADIENT = new DefaultGradient();

    //current input and label matrices
    @Setter @Getter
    protected INDArray labels;


    private double fullNetworkL1;
    private double fullNetworkL2;
    private double score;

    public Yolo2OutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray epsOut = computeBackpropGradientAndScore();

        return new Pair<>(EMPTY_GRADIENT, epsOut);
    }

    private INDArray computeBackpropGradientAndScore(){
        boolean DEBUG_PRINT = false;


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
        int c = labels.size(1)-4;

        INDArray wh = Nd4j.create(new double[]{w,h});
        INDArray boxPriors = layerConf().getBoundingBoxes();    //Shape: [b, 2]
        INDArray boxPriorsNormalized = boxPriors.divRowVector(wh); //pw * exp(in_w) and ph * exp(in_h)



        //Debugging code
        if(DEBUG_PRINT) {
            System.out.println("Class labels shape: " + Arrays.toString(classLabels.shape()));
            System.out.println("mb, h, w, b, c");
            System.out.println(mb + ", " + h + ", " + w + ", " + b + ", " + c);
        }

        // ----- Step 1: Labels format conversion -----
        //First: Convert labels/ground truth (x1,y1,x2,y2) from "number of grid boxes" format to center format, as
        // fraction of total image
        //0.5 * ([x1,y1]+[x2,y2])   ->      shape: [mb, 2, H, W]
        INDArray labelTLXYImg = labels.get(all(),interval(0,2), all(), all());
        INDArray labelBRXYImg = labels.get(all(),interval(2,4), all(), all());

        INDArray labelCenterXYImg = labelTLXYImg.add(labelBRXYImg).muli(0.5);
        Broadcast.div(labelCenterXYImg, wh, labelCenterXYImg, 1);


        //Then convert label centers from "fraction of total image" to "fraction of grid", which are used in position loss
        INDArray labelsCenterXYInGrid = Nd4j.createUninitialized(labelCenterXYImg.shape(), labelCenterXYImg.ordering());
        Broadcast.mul(labelCenterXYImg, wh, labelsCenterXYInGrid, 1 );
        labelsCenterXYInGrid.subi(Transforms.floor(labelsCenterXYInGrid,true));

        //Also infer size/scale (label w/h) from (x1,y1,x2,y2) format to (w,h) format
        INDArray labelWHSqrt = labelBRXYImg.sub(labelTLXYImg);
        labelWHSqrt = Transforms.sqrt(labelWHSqrt, true);



        // ----- Step 2: apply activation functions to network output activations -----
        //Reshape from [minibatch, 5B+C, H, W] to [minibatch, 5B, H, W] to [minibatch, B, 5, H, W]
        INDArray input5 = input.get(all(), interval(0,5*b), all(), all()).dup('c').reshape(mb, b, 5, h, w);
        INDArray inputClasses = input.get(all(), interval(5*b, 5*b+c), all(), all());

        // Sigmoid for x/y centers
        INDArray preSigmoidPredictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        INDArray predictedXYCenterGrid = Transforms.sigmoid(preSigmoidPredictedXYCenterGrid, true);

        //Exponential for w/h (for: boxPrior * exp(input))
        INDArray predictedWHPreExp = input5.get(all(), all(), interval(2,4), all(), all());
        INDArray predictedWH = Transforms.exp(predictedWHPreExp, true);
        Broadcast.mul(predictedWH, boxPriorsNormalized, predictedWH, 1, 2);  //Box priors: [b, 2]; predictedWH: [mb, b, 2, h, w]


        //Calculate predicted top/left and bottom/right in overall image
        //First: calculate top/left  value for each grid location. gridXY contains
        INDArray xVector = Nd4j.linspace(0, 1.0-1.0/w, w);  //[0 to w-1]/w
        INDArray yVector = Nd4j.linspace(0, 1.0-1.0/h, h);  //[0 to h-1]/h
        INDArray gridYX = Nd4j.create(2,h,w);
        gridYX.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).putiRowVector(xVector);
        gridYX.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).putiRowVector(yVector);

        INDArray predictedXYCenterImage = Nd4j.createUninitialized(predictedXYCenterGrid.shape(), predictedXYCenterGrid.ordering());
        Broadcast.div(predictedXYCenterGrid, wh, predictedXYCenterImage, 2 );   //[1,2] to [minibatch, B, 2, H, W]
        Broadcast.add(predictedXYCenterImage, gridYX, predictedXYCenterImage, 2,3,4); // [2,H,W] to [minibatch, B, 2, H, W]

        INDArray halfWidth = predictedWH.mul(0.5);
        INDArray predictedTLXYImage = predictedXYCenterImage.sub(halfWidth);
        INDArray predictedBRXYImage = halfWidth.addi(predictedXYCenterImage);

        //Apply sqrt to W/H in preparation for loss function
        INDArray predictedWHSqrt = Transforms.sqrt(predictedWH, true);




        // ----- Step 3: Calculate IOU(predicted, labels) to infer 1_ij^obj mask array (for loss function) -----
        //Calculate IOU (intersection over union - aka Jaccard index) - for the labels and predicted values
        IOURet iouRet = calculateIOULabelPredicted(labelTLXYImg, labelBRXYImg, predictedWH, predictedXYCenterGrid, maskArray);  //IOU shape: [minibatch, B, H, W]
        INDArray iou = iouRet.getIou();

        double iouMin = iou.minNumber().doubleValue();
        double iouMax = iou.maxNumber().doubleValue();
        if(iouMin < 0.0){
            throw new IllegalStateException("Invalid IOU: min value is " + iouMin);
        }

        if(iouMax > 1.0){
            throw new IllegalStateException("Invalid IOU: max value is " + iouMax);
        }

        //Mask 1_ij^obj: isMax (dimension 1) + apply object present mask. Result: [minibatch, B, H, W]
        //In this mask: 1 if (a) object is present in cell [for each mb/H/W], and (b) for the anchor box with the max IOU
        INDArray mask1_ij_obj = Nd4j.getExecutioner().execAndReturn(new IsMax(iou.dup(iou.ordering()), 1));
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(mask1_ij_obj, maskArray, mask1_ij_obj, 0,2,3));

        // ----- Step 4: Calculate confidence, and confidence label -----
        //Predicted confidence: sigmoid (0 to 1)
        //Label confidence: 0 if no object, IOU(predicted,actual) otherwise

        INDArray labelConfidence = iou.mul(mask1_ij_obj);  //OK to reuse IOU array here.   //Shape: [mb, B, H, W]
        INDArray predictedConfidencePreSigmoid = input5.get(all(), all(), point(4), all(), all());    //Shape: [mb, B, H, W]
        INDArray predictedConfidence = Transforms.sigmoid(predictedConfidencePreSigmoid, true);


        // ----- Step 5: Loss Function -----

        //One design goal here is to make the loss function configurable. To do this, we want to reshape the activations
        //(and masks) to a 2d representation, suitable for use in DL4J's loss functions

        INDArray mask1_ij_obj_2d = mask1_ij_obj.dup('c').reshape(mb*b*h*w, 1);
        INDArray mask1_ij_noobj_2d = Transforms.not(mask1_ij_obj_2d);   //Not op is copy op; mask has 1 where box is not responsible for prediction
//        INDArray mask2d = maskArray.dup('c').reshape(mb*h*w, 1);
        INDArray mask2d = maskArray.dup('c').reshape('c', new int[]{mb*h*w, 1});

        INDArray predictedXYCenter2d = predictedXYCenterGrid.permute(0,1,3,4,2)  //From: [mb, B, 2, H, W] to [mb, B, H, W, 2]
                .dup('c').reshape('c', mb*b*h*w, 2);        //TODO need permute first??
        /*
        //Don't use INDArray.broadcast(int...) until ND4J issue is fixed:
        // https://github.com/deeplearning4j/nd4j/issues/2066
        System.out.println(Arrays.toString(labelsCenterXYInGrid.shape()));
        INDArray labelsCenterXYInGridBroadcast = labelsCenterXYInGrid.broadcast(mb, b, 2, h, w);
        System.out.println(mb + "\t" + b + "\t" + h + "\t" + w);
        System.out.println(Arrays.toString(labelsCenterXYInGridBroadcast.shape()));
        */
        //Broadcast labelsCenterXYInGrid from [mb, 2, h, w} to [mb, b, 2, h, w]
        INDArray labelsCenterXYInGridBroadcast = Nd4j.createUninitialized(new int[]{mb, b, 2, h, w}, 'c');
        for(int i=0; i<b; i++ ){
            labelsCenterXYInGridBroadcast.get(all(), point(i), all(), all(), all()).assign(labelsCenterXYInGrid);
        }
        INDArray labelXYCenter2d = labelsCenterXYInGridBroadcast.permute(0,1,3,4,2).dup('c').reshape('c', mb*b*h*w, 2);    //[mb, b, 2, h, w] to [mb, b, h, w, 2] to [mb*b*h*w, 2]


        //Width/height (sqrt)
        INDArray predictedWHSqrt2d = predictedWHSqrt.permute(0,1,3,4,2).dup('c').reshape(mb*b*h*w, 2).dup('c'); //from [mb, b, 2, h, w] to [mb, b, h, w, 2] to [mb*b*h*w, 2]
        //Broadcast labelWHSqrt from [mb, 2, h, w} to [mb, b, 2, h, w]
        INDArray labelWHSqrtBroadcast = Nd4j.createUninitialized(new int[]{mb, b, 2, h, w}, 'c');
        for(int i=0; i<b; i++ ){
            labelWHSqrtBroadcast.get(all(), point(i), all(), all(), all()).assign(labelWHSqrt); //[mb, 2, h, w] to [mb, b, 2, h, w]
        }
        INDArray labelWHSqrt2d = labelWHSqrtBroadcast.permute(0,1,3,4,2).dup('c').reshape(mb*b*h*w, 2).dup('c');   //[mb, b, 2, h, w] to [mb, b, h, w, 2] to [mb*b*h*w, 2]

        //Confidence
        INDArray labelConfidence2d = labelConfidence.dup('c').reshape('c', mb * b * h * w, 1);
        INDArray predictedConfidence2d = predictedConfidence.dup('c').reshape('c', mb * b * h * w, 1).dup('c');
        INDArray predictedConfidence2dPreSigmoid = predictedConfidencePreSigmoid.dup('c').reshape('c', mb * b * h * w, 1).dup('c');


        //Class prediction loss
        INDArray classPredictionsPreSoftmax = inputClasses;      //Shape: [minibatch, C, H, W]
        INDArray classPredictionsPreSoftmax2d = classPredictionsPreSoftmax.permute(0,2,3,1) //To [mb, h, w, c]
                .dup('c').reshape('c', new int[]{mb*h*w, c});
        INDArray classLabels2d = classLabels.permute(0,2,3,1).dup('c').reshape('c', new int[]{mb*h*w, c});

        //Calculate the loss:
        IActivation identity = new ActivationIdentity();
        double positionLoss = layerConf().getLossPositionScale().computeScore(labelXYCenter2d, predictedXYCenter2d, identity, mask1_ij_obj_2d, false );
        double sizeScaleLoss = layerConf().getLossPositionScale().computeScore(labelWHSqrt2d, predictedWHSqrt2d, identity, mask1_ij_obj_2d, false);
        double confidenceLoss = layerConf().getLossConfidence().computeScore(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_obj_2d, false)
                + layerConf().getLambdaNoObj() * layerConf().getLossConfidence().computeScore(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_noobj_2d, false);    //TODO: possible to optimize this?

//        double confidenceLoss = layerConf().getLossConfidence().computeScore(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_obj_2d, false);
//                + layerConf().getLambdaNoObj() * layerConf().getLossConfidence().computeScore(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_noobj_2d, false);    //TODO: possible to optimize this?

        double classPredictionLoss = layerConf().getLossClassPredictions().computeScore(classLabels2d, classPredictionsPreSoftmax2d, new ActivationSoftmax(), mask2d, false);

        if(DEBUG_PRINT){
            System.out.println(Arrays.toString(labelConfidence2d.shape()) + "\t" + Arrays.toString(predictedConfidence2d.shape()));
            System.out.println("Label confidence: min/max - " + labelConfidence2d.minNumber() + "\t" + labelConfidence2d.maxNumber());
            System.out.println("label - confidence: ");
            System.out.println(labelConfidence2d);
            System.out.println("predicted - confidence: ");
            System.out.println(predictedConfidence2d);
        }

        //----------
        //DEBUGGING:
        if(DEBUG_PRINT) {
            System.out.println("position, size/scale, confidence, classPrediction");
            System.out.println(positionLoss + "\t" + sizeScaleLoss + "\t" + confidenceLoss + "\t" + classPredictionLoss);
        }
        //----------

//        double loss = layerConf().getLambdaCoord() * (positionLoss + sizeScaleLoss)
//                + confidenceLoss
//                + classPredictionLoss
//                + fullNetworkL1
//                + fullNetworkL2;

//        double loss = layerConf().getLambdaCoord() * (positionLoss + sizeScaleLoss)
//                + confidenceLoss
//                + classPredictionLoss
//                + fullNetworkL1
//                + fullNetworkL2;

        double loss =
                layerConf().getLambdaCoord() * positionLoss +
                layerConf().getLambdaCoord() * sizeScaleLoss +
                confidenceLoss +
                classPredictionLoss +
                fullNetworkL1 +
                fullNetworkL2
                ;

        loss /= getInputMiniBatchSize();

        this.score = loss;

        //===============================================
        // ----- Gradient Calculation (specifically: return dL/dIn -----

        if(DEBUG_PRINT) {
            System.out.println("Input shape: " + Arrays.toString(input.shape()));
        }
        INDArray epsOut = Nd4j.create(input.shape(), 'c');
        INDArray epsOut5 = Shape.newShapeNoCopy(epsOut.get(all(), interval(0,5*b), all(), all()), new int[]{mb, b, 5, h, w}, false);
        INDArray epsClassPredictions = epsOut.get(all(), interval(5*b, 5*b+c), all(), all());
        INDArray epsXY = epsOut5.get(all(), all(), interval(0,2), all(), all());
        INDArray epsWH = epsOut5.get(all(), all(), interval(2,4), all(), all());
        INDArray epsC = epsOut5.get(all(), all(), point(4), all(), all());


        //Calculate gradient component from class probabilities (softmax)
        //Shape: [minibatch*h*w, c]
        INDArray gradPredictionLoss2d = layerConf().getLossClassPredictions().computeGradient(classLabels2d, classPredictionsPreSoftmax2d, new ActivationSoftmax(), mask2d);
        INDArray gradPredictionLoss4d = gradPredictionLoss2d.dup('c').reshape(mb, h, w, c)
                .permute(0,3,1,2).dup('c');
        epsClassPredictions.assign(gradPredictionLoss4d);


        //Calculate gradient component from position (x,y) loss - dL_position/dx and dL_position/dy
        INDArray gradXYCenter2d = layerConf().getLossPositionScale().computeGradient(labelXYCenter2d, predictedXYCenter2d, identity, mask1_ij_obj_2d);
        gradXYCenter2d.muli(layerConf().getLambdaCoord());
        INDArray gradXYCenter5d = gradXYCenter2d.dup('c')
                .reshape(mb, b, h, w, 2)
                .permute(0,1,4,2,3)   //From: [mb, B, H, W, 2] to [mb, B, 2, H, W]
                .dup('c');
        gradXYCenter5d = new ActivationSigmoid().backprop(preSigmoidPredictedXYCenterGrid, gradXYCenter5d).getFirst();
        epsXY.assign(gradXYCenter5d);

        //Calculate gradient component from width/height (w,h) loss - dL_size/dw and dL_size/dw
        //Note that loss function gets sqrt(w) and sqrt(h)
        //gradWHSqrt2d = dL/dsqrt(w) and dL/dsqrt(h)
        INDArray gradWHSqrt2d = layerConf().getLossPositionScale().computeGradient(labelWHSqrt2d, predictedWHSqrt2d.dup(), identity, mask1_ij_obj_2d);   //Shape: [mb*b*h*w, 2]
            //dL/dw = dL/dsqrtw * dsqrtw / dw = dL/dsqrtw * 0.5 / sqrt(w)
        INDArray gradWH2d = gradWHSqrt2d.mul(0.5).divi(predictedWHSqrt2d);  //dL/dw and dL/dh, w = pw * exp(tw)
            //dL/dinWH = dL/dw * dw/dInWH = dL/dw * pw * exp(tw)
        INDArray gradWH5d = gradWH2d.dup('c').reshape(mb, b, h, w, 2).permute(0,1,4,2,3).dup('c');   //To: [mb, b, 2, h, w]
        gradWH5d.muli(predictedWH);
        gradWH5d.muli(layerConf().getLambdaCoord());
        epsWH.assign(gradWH5d);


        //Calculate gradient component from confidence loss... 2 parts (object present, no object present)
        //double confidenceLoss = layerConf().getLossConfidence().computeScore(labelConfidence2d, predictedCondidence2d, identity, mask1_ij_obj_2d, false)
        //  + layerConf().getLambdaNoObj() * layerConf().getLossConfidence().computeScore(labelConfidence2d, predictedCondidence2d, identity, mask1_ij_noobj_2d, false);    //TODO: possible to optimize this?

        ActivationSigmoid s = new ActivationSigmoid();
        INDArray gradConfidence2dA = layerConf().getLossConfidence().computeGradient(labelConfidence2d, predictedConfidence2d, s, mask1_ij_obj_2d);
        INDArray gradConfidence2dB = layerConf().getLossConfidence().computeGradient(labelConfidence2d, predictedConfidence2d, s, mask1_ij_noobj_2d);

        INDArray gradConfidence2d = gradConfidence2dA.addi(gradConfidence2dB);  //dL/dC; C = sigmoid(tc)
        //Calculate dL/dtc
        INDArray epsConfidence4d = gradConfidence2d.dup('c').reshape(mb, b, h, w);   //[mb*b*h*w, 2] to [mb, b, h, w]
        epsC.assign(epsConfidence4d);

        //Note that we ALSO have components to x,y,w,h  from confidence loss (via IOU)
        //that is: dL_conf/dx, dL_conf/dy, dL_conf/dw, dL_conf/dh
        //For any value v, d(I/U)/dv = (U * dI/dv + I * dU/dv) / U^2

        //Confidence loss: sum squared errors + masking.
        //C == IOU when label present
        INDArray dLc_dClabel = iou.sub(predictedConfidence).muli(2.0);  //Shape: [mb, b, h, w]
//        dLc_dClabel.mul(mask1_ij_obj.add(mask1_ij_noobj));
        INDArray newMask = mask1_ij_obj.dup();
        BooleanIndexing.applyWhere(newMask, Conditions.equals(0), layerConf().getLambdaNoObj());    //1 or lambda, for object or no-object respectively
        dLc_dClabel.mul(newMask);


        INDArray dLc_dIOU = dLc_dClabel;    //TODO - dL_C / dIOU, shape [mb, b, h, w]

        INDArray u = iouRet.getUnion();
        INDArray i = iouRet.getIntersection();
        INDArray u2 = iouRet.getUnion().mul(iouRet.getUnion());

        INDArray iuDivU2 = u.add(i).divi(u2);   //Shape: [mb, b, h, w]
        BooleanIndexing.replaceWhere(iuDivU2, 0.0, Conditions.isNan());     //Handle 0/0



        INDArray dIOU_dxy = Nd4j.createUninitialized(new int[]{mb, b, 2, h, w}, 'c');
        Broadcast.mul(iouRet.dIdxy_predicted, iuDivU2, dIOU_dxy, 0, 1, 3, 4);   //[mb, b, h, w] x [mb, b, 2, h, w]
        INDArray dLcdxy = Nd4j.createUninitialized(dIOU_dxy.shape(), dIOU_dxy.ordering());
        Broadcast.mul(dIOU_dxy, dLc_dIOU, dLcdxy, 0, 1, 3, 4);

        INDArray uSubI = u.sub(i);  //Shape: [mb, b, h, w]

        INDArray Iwh = Nd4j.createUninitialized(predictedWH.shape(), predictedWH.ordering());
        Broadcast.mul(predictedWH, iouRet.getIntersection(), Iwh, 0, 1, 3, 4 );    //Predicted_wh: [mb, b, 2, h, w]; intersection: [mb, b, h, w]
        INDArray dIOU_dwh = Nd4j.createUninitialized(new int[]{mb, b, 2, h, w}, iouRet.dIdwh_predicted.ordering());    //iouRet.dIdwh_predicted.mul(uSubI).add(Iwh).div(u2);
        Broadcast.mul(iouRet.dIdwh_predicted, uSubI, dIOU_dwh, 0, 1, 3, 4);
        Broadcast.div(dIOU_dwh, u2, dIOU_dwh, 0, 1, 3, 4);
        BooleanIndexing.replaceWhere(dIOU_dwh, 0.0, Conditions.isNan());     //Handle division by 0 (due to masking, etc)

        INDArray dLc_dwh = Nd4j.createUninitialized(dIOU_dwh.shape(), dIOU_dwh.ordering());
        INDArray dLc_dxy = Nd4j.createUninitialized(dIOU_dxy.shape(), dIOU_dxy.ordering());
        Broadcast.mul(dIOU_dwh, dLc_dIOU, dLc_dwh, 0, 1, 3, 4);    //[mb, b, h, w] x [mb, b, 2, h, w]
        Broadcast.mul(dIOU_dxy, dLc_dIOU, dLc_dxy, 0, 1, 3, 4);


        //Backprop through the wh and xy activation functions...
        INDArray labelWH = labelBRXYImg.sub(labelTLXYImg);
        //dL/dw and dL/dh, w = pw * exp(tw), //dL/dinWH = dL/dw * dw/dInWH = dL/dw * pw * exp(tw)
        INDArray dLc_din_wh = Broadcast.mul(dLc_dwh, labelWH, dLc_dwh, 0, 2, 3);    //[mb, h, w] x [mb, b, 2, h, w]
        INDArray dLc_din_xy = new ActivationSigmoid().backprop(preSigmoidPredictedXYCenterGrid, dLc_dxy).getFirst();


        epsWH.addi(dLc_din_wh);
        epsXY.addi(dLc_din_xy);

        return epsOut;
    }

    @Override
    public INDArray activationMean() {
        return activate();
    }

    @Override
    public INDArray activate(boolean training) {
        //Essentially: just apply activation functions...

        int mb = input.size(0);
        int h = input.size(2);
        int w = input.size(3);
        int b = layerConf().getBoundingBoxes().size(0);
        int c = input.size(1)-5*b;

        INDArray input5 = input.get(all(), interval(0,5*b), all(), all()).dup('c').reshape(mb, b, 5, h, w);

        //X/Y center in grid: sigmoid
        INDArray predictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        Transforms.sigmoid(predictedXYCenterGrid, false);

        //width/height: prior * exp(input)
        INDArray predictedWH = input5.get(all(), all(), interval(2,4), all(), all());   //Shape: [mb, B, 2, H, W]
        Transforms.exp(predictedWH, false);
        INDArray priorBoxes = layerConf().getBoundingBoxes();   //Shape: [B, 2]
        Broadcast.mul(predictedWH, priorBoxes, predictedWH, 1, 2);

        //Confidence - sigmoid
        INDArray predictedConf = input5.get(all(), all(), point(4), all(), all());   //Shape: [mb, B, H, W]
        Transforms.sigmoid(predictedConf, false);


        //Softmax
        //TODO OPTIMIZE
        INDArray inputClasses = input.get(all(), interval(5*b, 5*b+c), all(), all());   //Shape: [minibatch, C, H, W]
        INDArray classPredictionsPreSoftmax2d = inputClasses.permute(0,2,3,1).dup('c')  //Shape before reshape: [mb, h, w, c]
                .reshape(mb*h*w, c);
        Transforms.softmax(classPredictionsPreSoftmax2d);
        INDArray postSoftmax4d = classPredictionsPreSoftmax2d.reshape('c', mb, h, w, c ).permute(0, 3, 1, 2);
        inputClasses.assign(postSoftmax4d);

        return input;
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training) {
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;

        //TODO optimize?
        computeBackpropGradientAndScore();
        return score();
    }

    @Override
    public double score(){
        return score;
    }

    /**
     * Calculate IOU(truth, predicted). Returns 5d array, [mb, b, 2, H, W]
     *
     * @param labelTL   4d [mb, 2, H, W], label top/left (x,y) in terms of grid boxes
     * @param predictedWH 5d [mb, b, 2, H, W] - predicted H/W in terms of number of grid boxes.
     * @param predictedXY 5d [mb, b, 2, H, W] - predicted X/Y in terms of number of grid boxes. Values 0 to 1, center box value being 0.5
     * @return
     */
    private static IOURet calculateIOULabelPredicted(INDArray labelTL, INDArray labelBR, INDArray predictedWH, INDArray predictedXY, INDArray objectPresentMask){

        double minwh = predictedWH.minNumber().doubleValue();
        if(minwh < 0.0){
            throw new IllegalStateException("Min width/height: " + minwh);
        }

        //Predicted x/y shouldbe in 0 to 1 (due to sigmoid - it is position in grid...)
        double minPredXY = predictedXY.minNumber().doubleValue();
        double maxPredXY = predictedXY.maxNumber().doubleValue();
        if(minPredXY < 0.0){
            throw new IllegalStateException("Min predicted XY: " + minPredXY);
        }
        if(maxPredXY > 1.0){
            throw new IllegalStateException("Max predicted XY: " + minPredXY);
        }

        INDArray labelWH = labelBR.sub(labelTL);                //4d [mb, 2, H, W], label W/H in terms of number of grid boxes

        INDArray halfWH = predictedWH.mul(0.5);
        INDArray predictedTL_XY = halfWH.rsub(predictedXY);     //xy - 0.5 * wh
        INDArray predictedBR_XY = halfWH.add(predictedXY);      //xy + 0.5 * wh


        INDArray maxTL = Nd4j.createUninitialized(predictedTL_XY.shape(), predictedTL_XY.ordering());   //Shape: [mb, b, 2, H, W]
        Broadcast.max(predictedTL_XY, labelTL, maxTL, 0, 2, 3, 4);
        INDArray minBR = Nd4j.createUninitialized(predictedBR_XY.shape(), predictedBR_XY.ordering());
        Broadcast.min(predictedBR_XY, labelBR, minBR, 0, 2, 3, 4);

        INDArray diff = minBR.sub(maxTL);
        INDArray intersectionArea = diff.prod(2);   //[mb, b, 2, H, W] to [mb, b, H, W]
        Broadcast.mul(intersectionArea, objectPresentMask, intersectionArea, 0, 2, 3);

        //Need to mask the calculated intersection values, to avoid returning non-zero values for 0 intersection
        //No intersection if: xP + wP/2 < xL - wL/2 i.e., BR_xPred < TL_xLab   OR  TL_xPred > BR_xLab (similar for Y axis)
        //Here, 1 if intersection exists, 0 otherwise. This is doing x/w and y/h simultaneously

        INDArray noIntMask1 = Nd4j.create(maxTL.shape(), maxTL.ordering());
        INDArray noIntMask2 = Nd4j.create(maxTL.shape(), maxTL.ordering());
        //Does both x and y on different dims
        Broadcast.lt(predictedBR_XY, labelTL, noIntMask1, 0, 2, 3, 4);  //Predicted BR < label TL
        Broadcast.gt(predictedTL_XY, labelBR, noIntMask2, 0, 2, 3, 4);  //predicted TL > label BR

//        INDArray noIntMask = noIntMask1.prod(2).muli(noIntMask2.prod(2));   //Shape: [mb, b, H, W]. Values 1 if no intersection
        INDArray noIntMask = Transforms.or(noIntMask1.get(all(), all(), point(0), all(), all()), noIntMask1.get(all(), all(), point(1), all(), all()) );
        Transforms.or(noIntMask2.get(all(), all(), point(0), all(), all()), noIntMask2.get(all(), all(), point(1), all(), all()) );
        Transforms.or(noIntMask1, noIntMask2 );

        INDArray intMask = Nd4j.getExecutioner().execAndReturn(new Not(noIntMask, noIntMask, 0.0)); //Values 0 if no intersection
        Broadcast.mul(intMask, objectPresentMask, intMask, 0, 2, 3);

        //Mask the intersection area: should be 0 if no intersection
        intersectionArea.muli(intMask);

        int totalCount = intMask.length();
        int countOne = intMask.sumNumber().intValue();
//        System.out.println("intMask counts: total, number of 1s: " + totalCount + ", " + countOne);

        double minIntArea = intersectionArea.minNumber().doubleValue();
        double maxIntArea = intersectionArea.maxNumber().doubleValue();

        if(minIntArea < 0.0){
            throw new IllegalStateException("Min intersection area: " + minIntArea);
        }

        if(maxIntArea < 0.0){
            throw new IllegalStateException("max intersection area: " + maxIntArea);
        }

        //*** deBUG ***
        Broadcast.mul(predictedWH, objectPresentMask, predictedWH, 0,3,4);


        //Next, union area is simple: U = A1 + A2 - intersection
        INDArray areaPredicted = predictedWH.prod(2);   //[mb, b, 2, H, W] to [mb, b, H, W]
        Broadcast.mul(areaPredicted, objectPresentMask, areaPredicted, 0,2,3);
        INDArray areaLabel = labelWH.prod(1);           //[mb, 2, H, W] to [mb, H, W]

        INDArray unionArea = Broadcast.add(areaPredicted, areaLabel, areaPredicted.dup(), 0, 2, 3);
        unionArea.subi(intersectionArea);
        unionArea.muli(intMask);

        double minUnion = unionArea.minNumber().doubleValue();
        double maxUnion = unionArea.maxNumber().doubleValue();
        if(minUnion < 0.0){
            throw new IllegalStateException("Min union area: " + minUnion);
        }

        INDArray iou = intersectionArea.div(unionArea);
        BooleanIndexing.replaceWhere(iou, 0.0, Conditions.isNan()); //0/0 -> NaN -> 0

        //Apply the "object present" mask (of shape [mb, h, w]
        Broadcast.mul(iou, objectPresentMask, iou, 0, 2, 3);


        double minIou = iou.minNumber().doubleValue();
        double maxIou = iou.maxNumber().doubleValue();

        if(minIou < 0 ){
            throw new IllegalStateException("Min IOU: " + minIou);
        }
        if(maxIou < 0 || maxIou > 1.0 ){
            throw new IllegalStateException("Max IOU: " + maxIou);
        }

        //Finally, calculate derivatives:
        INDArray maskMaxTL = Nd4j.create(maxTL.shape(), maxTL.ordering());    //1 if predicted Top/Left is max, 0 otherwise
        Broadcast.gt(predictedTL_XY, labelTL, maskMaxTL, 0, 2, 3, 4);   // z = x > y

        INDArray maskMinBR = Nd4j.create(maxTL.shape(), maxTL.ordering());    //1 if predicted Top/Left is max, 0 otherwise
        Broadcast.lt(predictedBR_XY, labelBR, maskMinBR, 0, 2, 3, 4);   // z = x < y

        INDArray dIdxy_predicted = maskMinBR.sub(maskMaxTL);
        INDArray dIdwh_predicted = maskMinBR.add(maskMaxTL).muli(0.5);

        return new IOURet(iou, intersectionArea, unionArea, dIdxy_predicted, dIdwh_predicted);
    }


    @AllArgsConstructor
    @Data
    private static class IOURet {
        private INDArray iou;
        private INDArray intersection;
        private INDArray union;
        private INDArray dIdxy_predicted;
        private INDArray dIdwh_predicted;

    }

    @Override
    public void computeGradientAndScore(){
        //Assume full network l1/l2 is already provided...

        //TODO
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray preOutput(boolean training) {
        return input;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2) {
        throw new UnsupportedOperationException();
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
