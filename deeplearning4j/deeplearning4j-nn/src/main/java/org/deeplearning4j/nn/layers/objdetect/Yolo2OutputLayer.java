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

package org.deeplearning4j.nn.layers.objdetect;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Not;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Output (loss) layer for YOLOv2 object detection model, based on the papers:
 * YOLO9000: Better, Faster, Stronger - Redmon & Farhadi (2016) - <a href="https://arxiv.org/abs/1612.08242">https://arxiv.org/abs/1612.08242</a><br>
 * and<br>
 * You Only Look Once: Unified, Real-Time Object Detection - Redmon et al. (2016) -
 * <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf">http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf</a>
 * <br>
 * This loss function implementation is based on the YOLOv2 version of the paper. However, note that it doesn't
 * currently support simultaneous training on both detection and classification datasets as described in the
 * YOlO9000 paper.<br>
 * <br>
 * Label format: [minibatch, 4+C, H, W]<br>
 * Order for labels depth: [x1,y1,x2,y2,(class labels)]<br>
 * x1 = box top left position<br>
 * y1 = as above, y axis<br>
 * x2 = box bottom right position<br>
 * y2 = as above y axis<br>
 * Note: labels are represented as a multiple of grid size - for a 13x13 grid, (0,0) is top left, (13,13) is bottom right<br>
 * <br>
 * Input format: [minibatch, B*(5+C), H, W]    ->      Reshape to [minibatch, B, 5+C, H, W]<br>
 * B = number of bounding boxes (determined by config)<br>
 * C = number of classes<br>
 * H = output/label height<br>
 * W = output/label width<br>
 * <br>
 * Note that mask arrays are not required - this implementation infers the presence or absence of objects in each grid
 * cell from the class labels (which should be 1-hot if an object is present, or all 0s otherwise).
 *
 * @author Alex Black
 */
public class Yolo2OutputLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer> implements Serializable, IOutputLayer {
    private static final Gradient EMPTY_GRADIENT = new DefaultGradient();

    //current input and label matrices
    @Setter @Getter
    protected INDArray labels;

    private double fullNetRegTerm;
    private double score;

    public Yolo2OutputLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        INDArray epsOut = computeBackpropGradientAndScore(workspaceMgr, false, false);

        return new Pair<>(EMPTY_GRADIENT, epsOut);
    }

    private INDArray computeBackpropGradientAndScore(LayerWorkspaceMgr workspaceMgr, boolean scoreOnly, boolean computeScoreForExamples){
        assertInputSet(true);
        Preconditions.checkState(labels != null, "Cannot calculate gradients/score: labels are null");
        Preconditions.checkState(labels.rank() == 4, "Expected rank 4 labels array with shape [minibatch, 4+numClasses, h, w]" +
                " but got rank %s labels array with shape %s", labels.rank(), labels.shape());

        double lambdaCoord = layerConf().getLambdaCoord();
        double lambdaNoObj = layerConf().getLambdaNoObj();

        long mb = input.size(0);
        long h = input.size(2);
        long w = input.size(3);
        int b = (int) layerConf().getBoundingBoxes().size(0);
        int c = (int) labels.size(1)-4;

        INDArray labels = this.labels.castTo(input.dataType());     //Ensure correct dtype (same as params); no-op if already correct dtype

        //Various shape arrays, to reuse
        long[] nhw = new long[]{mb, h, w};

        //Labels shape: [mb, 4+C, H, W]
        //Infer mask array from labels. Mask array is 1_i^B in YOLO paper - i.e., whether an object is present in that
        // grid location or not. Here: we are using the fact that class labels are one-hot, and assume that values are
        // all 0s if no class label is present
        Preconditions.checkState(labels.rank() == 4, "Expected labels array to be rank 4 with shape [minibatch, 4+numClasses, H, W]. Got labels array with shape %ndShape", labels);
        Preconditions.checkState(labels.size(1) > 0, "Invalid labels array: labels.size(1) must be > 4. labels array should be rank 4 with shape [minibatch, 4+numClasses, H, W]. Got labels array with shape %ndShape", labels);

        val size1 = labels.size(1);
        INDArray classLabels = labels.get(all(), interval(4,size1), all(), all());   //Shape: [minibatch, nClasses, H, W]
        INDArray maskObjectPresent = classLabels.sum(Nd4j.createUninitialized(input.dataType(), nhw, 'c'), 1);//.castTo(DataType.BOOL); //Shape: [minibatch, H, W]
        INDArray maskObjectPresentBool = maskObjectPresent.castTo(DataType.BOOL);

        // ----- Step 1: Labels format conversion -----
        //First: Convert labels/ground truth (x1,y1,x2,y2) from "coordinates (grid box units)" format to "center position in grid box" format
        //0.5 * ([x1,y1]+[x2,y2])   ->      shape: [mb, B, 2, H, W]
        INDArray labelTLXY = labels.get(all(), interval(0,2), all(), all());
        INDArray labelBRXY = labels.get(all(), interval(2,4), all(), all());

        INDArray labelCenterXY = labelTLXY.add(labelBRXY).muli(0.5);  //In terms of grid units
        INDArray labelsCenterXYInGridBox = labelCenterXY.dup(labelCenterXY.ordering());         //[mb, 2, H, W]
        labelsCenterXYInGridBox.subi(Transforms.floor(labelsCenterXYInGridBox,true));

        //Also infer size/scale (label w/h) from (x1,y1,x2,y2) format to (w,h) format
        INDArray labelWHSqrt = labelBRXY.sub(labelTLXY);
        labelWHSqrt = Transforms.sqrt(labelWHSqrt, false);



        // ----- Step 2: apply activation functions to network output activations -----
        //Reshape from [minibatch, B*(5+C), H, W] to [minibatch, B, 5+C, H, W]
        long[] expInputShape = new long[]{mb, b*(5+c), h, w};
        long[] newShape = new long[]{mb, b, 5+c, h, w};
        long newLength = ArrayUtil.prodLong(newShape);
        Preconditions.checkState(Arrays.equals(expInputShape, input.shape()), "Unable to reshape input - input array shape does not match" +
                " expected shape. Expected input shape [minibatch, B*(5+C), H, W]=%s but got input of shape %ndShape. This may be due to an incorrect nOut (layer size/channels)" +
                " for the last convolutional layer in the network. nOut of the last layer must be B*(5+C) where B is the number of" +
                " bounding boxes, and C is the number of object classes. Expected B=%s from network configuration and C=%s from labels", expInputShape, input, b, c);
        INDArray input5 = input.dup('c').reshape('c', mb, b, 5+c, h, w);
        INDArray inputClassesPreSoftmax = input5.get(all(), all(), interval(5, 5+c), all(), all());

        // Sigmoid for x/y centers
        INDArray preSigmoidPredictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        INDArray predictedXYCenterGrid = Transforms.sigmoid(preSigmoidPredictedXYCenterGrid, true); //Not in-place, need pre-sigmoid later

        //Exponential for w/h (for: boxPrior * exp(input))      ->      Predicted WH in grid units (0 to 13 usually)
        INDArray predictedWHPreExp = input5.get(all(), all(), interval(2,4), all(), all());
        INDArray predictedWH = Transforms.exp(predictedWHPreExp, true);
        Broadcast.mul(predictedWH, layerConf().getBoundingBoxes().castTo(predictedWH.dataType()), predictedWH, 1, 2);  //Box priors: [b, 2]; predictedWH: [mb, b, 2, h, w]

        //Apply sqrt to W/H in preparation for loss function
        INDArray predictedWHSqrt = Transforms.sqrt(predictedWH, true);



        // ----- Step 3: Calculate IOU(predicted, labels) to infer 1_ij^obj mask array (for loss function) -----
        //Calculate IOU (intersection over union - aka Jaccard index) - for the labels and predicted values
        IOURet iouRet = calculateIOULabelPredicted(labelTLXY, labelBRXY, predictedWH, predictedXYCenterGrid, maskObjectPresent, maskObjectPresentBool);  //IOU shape: [minibatch, B, H, W]
        INDArray iou = iouRet.getIou();

        //Mask 1_ij^obj: isMax (dimension 1) + apply object present mask. Result: [minibatch, B, H, W]
        //In this mask: 1 if (a) object is present in cell [for each mb/H/W], AND (b) it is the box with the highest
        // IOU of any in the grid cell
        //We also need 1_ij^noobj, which is (a) no object, or (b) object present in grid cell, but this box doesn't
        // have the highest IOU
        INDArray mask1_ij_obj = Nd4j.create(DataType.BOOL, iou.shape(), 'c');
        Nd4j.exec(new IsMax(iou, mask1_ij_obj, 1));
        Nd4j.exec(new BroadcastMulOp(mask1_ij_obj, maskObjectPresentBool, mask1_ij_obj, 0,2,3));
        INDArray mask1_ij_noobj = Transforms.not(mask1_ij_obj);
        mask1_ij_obj = mask1_ij_obj.castTo(input.dataType());



        // ----- Step 4: Calculate confidence, and confidence label -----
        //Predicted confidence: sigmoid (0 to 1)
        //Label confidence: 0 if no object, IOU(predicted,actual) if an object is present
        INDArray labelConfidence = iou.mul(mask1_ij_obj);  //Need to reuse IOU array later. IOU Shape: [mb, B, H, W]
        INDArray predictedConfidencePreSigmoid = input5.get(all(), all(), point(4), all(), all());    //Shape: [mb, B, H, W]
        INDArray predictedConfidence = Transforms.sigmoid(predictedConfidencePreSigmoid, true);



        // ----- Step 5: Loss Function -----
        //One design goal here is to make the loss function configurable. To do this, we want to reshape the activations
        //(and masks) to a 2d representation, suitable for use in DL4J's loss functions

        INDArray mask1_ij_obj_2d = mask1_ij_obj.reshape(mb*b*h*w, 1);  //Must be C order before reshaping
        INDArray mask1_ij_noobj_2d = mask1_ij_obj_2d.rsub(1.0);


        INDArray predictedXYCenter2d = predictedXYCenterGrid.permute(0,1,3,4,2)  //From: [mb, B, 2, H, W] to [mb, B, H, W, 2]
                .dup('c').reshape('c', mb*b*h*w, 2);
        //Don't use INDArray.broadcast(int...) until ND4J issue is fixed: https://github.com/deeplearning4j/nd4j/issues/2066
        //INDArray labelsCenterXYInGridBroadcast = labelsCenterXYInGrid.broadcast(mb, b, 2, h, w);
        //Broadcast labelsCenterXYInGrid from [mb, 2, h, w} to [mb, b, 2, h, w]
        INDArray labelsCenterXYInGridBroadcast = Nd4j.createUninitialized(input.dataType(), new long[]{mb, b, 2, h, w}, 'c');
        for(int i=0; i<b; i++ ){
            labelsCenterXYInGridBroadcast.get(all(), point(i), all(), all(), all()).assign(labelsCenterXYInGridBox);
        }
        INDArray labelXYCenter2d = labelsCenterXYInGridBroadcast.permute(0,1,3,4,2).dup('c').reshape('c', mb*b*h*w, 2);    //[mb, b, 2, h, w] to [mb, b, h, w, 2] to [mb*b*h*w, 2]

        //Width/height (sqrt)
        INDArray predictedWHSqrt2d = predictedWHSqrt.permute(0,1,3,4,2).dup('c').reshape(mb*b*h*w, 2).dup('c'); //from [mb, b, 2, h, w] to [mb, b, h, w, 2] to [mb*b*h*w, 2]
        //Broadcast labelWHSqrt from [mb, 2, h, w} to [mb, b, 2, h, w]
        INDArray labelWHSqrtBroadcast = Nd4j.createUninitialized(input.dataType(), new long[]{mb, b, 2, h, w}, 'c');
        for(int i=0; i<b; i++ ){
            labelWHSqrtBroadcast.get(all(), point(i), all(), all(), all()).assign(labelWHSqrt); //[mb, 2, h, w] to [mb, b, 2, h, w]
        }
        INDArray labelWHSqrt2d = labelWHSqrtBroadcast.permute(0,1,3,4,2).dup('c').reshape(mb*b*h*w, 2).dup('c');   //[mb, b, 2, h, w] to [mb, b, h, w, 2] to [mb*b*h*w, 2]

        //Confidence
        INDArray labelConfidence2d = labelConfidence.dup('c').reshape('c', mb * b * h * w, 1);
        INDArray predictedConfidence2d = predictedConfidence.dup('c').reshape('c', mb * b * h * w, 1).dup('c');
        INDArray predictedConfidence2dPreSigmoid = predictedConfidencePreSigmoid.dup('c').reshape('c', mb * b * h * w, 1).dup('c');


        //Class prediction loss
        INDArray classPredictionsPreSoftmax2d = inputClassesPreSoftmax.permute(0,1,3,4,2) //[minibatch, b, c, h, w] To [mb, b, h, w, c]
                .dup('c').reshape('c', new long[]{mb*b*h*w, c});
        INDArray classLabelsBroadcast = Nd4j.createUninitialized(input.dataType(), new long[]{mb, b, c, h, w}, 'c');
        for(int i=0; i<b; i++ ){
            classLabelsBroadcast.get(all(), point(i), all(), all(), all()).assign(classLabels); //[mb, c, h, w] to [mb, b, c, h, w]
        }
        INDArray classLabels2d = classLabelsBroadcast.permute(0,1,3,4,2).dup('c').reshape('c', new long[]{mb*b*h*w, c});

        //Calculate the loss:
        ILossFunction lossConfidence = new LossL2();
        IActivation identity = new ActivationIdentity();


        if(computeScoreForExamples){
            INDArray positionLoss = layerConf().getLossPositionScale().computeScoreArray(labelXYCenter2d, predictedXYCenter2d, identity, mask1_ij_obj_2d );
            INDArray sizeScaleLoss = layerConf().getLossPositionScale().computeScoreArray(labelWHSqrt2d, predictedWHSqrt2d, identity, mask1_ij_obj_2d);
            INDArray confidenceLossPt1 = lossConfidence.computeScoreArray(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_obj_2d);
            INDArray confidenceLossPt2 = lossConfidence.computeScoreArray(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_noobj_2d).muli(lambdaNoObj);
            INDArray classPredictionLoss = layerConf().getLossClassPredictions().computeScoreArray(classLabels2d, classPredictionsPreSoftmax2d, new ActivationSoftmax(), mask1_ij_obj_2d);

            INDArray scoreForExamples = positionLoss.addi(sizeScaleLoss).muli(lambdaCoord)
                    .addi(confidenceLossPt1).addi(confidenceLossPt2.muli(lambdaNoObj))
                    .addi(classPredictionLoss)
                    .dup('c');

            scoreForExamples = scoreForExamples.reshape('c', mb, b*h*w).sum(true, 1);
            if(fullNetRegTerm > 0.0) {
                scoreForExamples.addi(fullNetRegTerm);
            }

            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, scoreForExamples);
        }

        double positionLoss = layerConf().getLossPositionScale().computeScore(labelXYCenter2d, predictedXYCenter2d, identity, mask1_ij_obj_2d, false );
        double sizeScaleLoss = layerConf().getLossPositionScale().computeScore(labelWHSqrt2d, predictedWHSqrt2d, identity, mask1_ij_obj_2d, false);
        double confidenceLoss = lossConfidence.computeScore(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_obj_2d, false)
                + lambdaNoObj * lossConfidence.computeScore(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_noobj_2d, false);    //TODO: possible to optimize this?
        double classPredictionLoss = layerConf().getLossClassPredictions().computeScore(classLabels2d, classPredictionsPreSoftmax2d, new ActivationSoftmax(), mask1_ij_obj_2d, false);

        this.score = lambdaCoord * (positionLoss + sizeScaleLoss) +
                confidenceLoss  +
                classPredictionLoss;

        this.score /= getInputMiniBatchSize();

        this.score += fullNetRegTerm;

        if(scoreOnly)
            return null;


        //==============================================================
        // ----- Gradient Calculation (specifically: return dL/dIn -----

        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape(), 'c');
        INDArray epsOut5 = Shape.newShapeNoCopy(epsOut, new long[]{mb, b, 5+c, h, w}, false);
        INDArray epsClassPredictions = epsOut5.get(all(), all(), interval(5, 5+c), all(), all());    //Shape: [mb, b, 5+c, h, w]
        INDArray epsXY = epsOut5.get(all(), all(), interval(0,2), all(), all());
        INDArray epsWH = epsOut5.get(all(), all(), interval(2,4), all(), all());
        INDArray epsC = epsOut5.get(all(), all(), point(4), all(), all());


        //Calculate gradient component from class probabilities (softmax)
        //Shape: [minibatch*h*w, c]
        INDArray gradPredictionLoss2d = layerConf().getLossClassPredictions().computeGradient(classLabels2d, classPredictionsPreSoftmax2d, new ActivationSoftmax(), mask1_ij_obj_2d);
        INDArray gradPredictionLoss5d = gradPredictionLoss2d.dup('c').reshape(mb, b, h, w, c).permute(0,1,4,2,3).dup('c');
        epsClassPredictions.assign(gradPredictionLoss5d);


        //Calculate gradient component from position (x,y) loss - dL_position/dx and dL_position/dy
        INDArray gradXYCenter2d = layerConf().getLossPositionScale().computeGradient(labelXYCenter2d, predictedXYCenter2d, identity, mask1_ij_obj_2d);
        gradXYCenter2d.muli(lambdaCoord);
        INDArray gradXYCenter5d = gradXYCenter2d.dup('c')
                .reshape('c', mb, b, h, w, 2)
                .permute(0,1,4,2,3);   //From: [mb, B, H, W, 2] to [mb, B, 2, H, W]
        gradXYCenter5d = new ActivationSigmoid().backprop(preSigmoidPredictedXYCenterGrid.dup(), gradXYCenter5d).getFirst();
        epsXY.assign(gradXYCenter5d);

        //Calculate gradient component from width/height (w,h) loss - dL_size/dW and dL_size/dW
        //Note that loss function gets sqrt(w) and sqrt(h)
        //gradWHSqrt2d = dL/dsqrt(w) and dL/dsqrt(h)
        INDArray gradWHSqrt2d = layerConf().getLossPositionScale().computeGradient(labelWHSqrt2d, predictedWHSqrt2d, identity, mask1_ij_obj_2d);   //Shape: [mb*b*h*w, 2]
            //dL/dW = dL/dsqrtw * dsqrtw / dW = dL/dsqrtw * 0.5 / sqrt(w)
        INDArray gradWH2d = gradWHSqrt2d.muli(0.5).divi(predictedWHSqrt2d);  //dL/dW and dL/dH, w = pw * exp(tw)
            //dL/dinWH = dL/dW * dW/dInWH = dL/dW * pw * exp(tw)
        INDArray gradWH5d = gradWH2d.dup('c').reshape(mb, b, h, w, 2).permute(0,1,4,2,3);   //To: [mb, b, 2, h, w]
        gradWH5d.muli(predictedWH);
        gradWH5d.muli(lambdaCoord);
        epsWH.assign(gradWH5d);


        //Calculate gradient component from confidence loss... 2 parts (object present, no object present)
        INDArray gradConfidence2dA = lossConfidence.computeGradient(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_obj_2d);
        INDArray gradConfidence2dB = lossConfidence.computeGradient(labelConfidence2d, predictedConfidence2d, identity, mask1_ij_noobj_2d);


        INDArray dLc_dC_2d = gradConfidence2dA.addi(gradConfidence2dB.muli(lambdaNoObj));  //dL/dC; C = sigmoid(tc)
        INDArray dLc_dzc_2d = new ActivationSigmoid().backprop( predictedConfidence2dPreSigmoid, dLc_dC_2d).getFirst();
        //Calculate dL/dtc
        INDArray epsConfidence4d = dLc_dzc_2d.dup('c').reshape('c', mb, b, h, w);   //[mb*b*h*w, 2] to [mb, b, h, w]
        epsC.assign(epsConfidence4d);





        //Note that we ALSO have components to x,y,w,h  from confidence loss (via IOU, which depends on all of these values)
        //that is: dLc/dx, dLc/dy, dLc/dW, dLc/dH
        //For any value v, d(I/U)/dv = (U * dI/dv + I * dU/dv) / U^2

        //Confidence loss: sum squared errors + masking.
        //C == IOU when label present

        //Lc = 1^(obj)*(iou - predicted)^2 + lambdaNoObj * 1^(noobj) * (iou - predicted)^2 -> dLc/diou = 2*1^(obj)*(iou-predicted) + 2 * lambdaNoObj * 1^(noobj) * (iou-predicted) = 2*(iou-predicted) * (1^(obj) + lambdaNoObj * 1^(noobj))
        INDArray twoIOUSubPredicted = iou.subi(predictedConfidence).muli(2.0);  //Shape: [mb, b, h, w]. Note that when an object is present, IOU and confidence are the same. In-place to avoid copy op (iou no longer needed)
        INDArray dLc_dIOU = twoIOUSubPredicted.muli(mask1_ij_noobj.castTo(input.dataType()).muli(lambdaNoObj).addi(mask1_ij_obj));


        INDArray dLc_dxy = Nd4j.createUninitialized(iouRet.dIOU_dxy.dataType(), iouRet.dIOU_dxy.shape(), iouRet.dIOU_dxy.ordering());
        Broadcast.mul(iouRet.dIOU_dxy, dLc_dIOU, dLc_dxy, 0, 1, 3, 4);    //[mb, b, h, w] x [mb, b, 2, h, w]

        INDArray dLc_dwh = Nd4j.createUninitialized(iouRet.dIOU_dwh.dataType(), iouRet.dIOU_dwh.shape(), iouRet.dIOU_dwh.ordering());
        Broadcast.mul(iouRet.dIOU_dwh, dLc_dIOU, dLc_dwh, 0, 1, 3, 4);    //[mb, b, h, w] x [mb, b, 2, h, w]


        //Backprop through the wh and xy activation functions...
        //dL/dW and dL/dH, w = pw * exp(tw), //dL/dinWH = dL/dW * dW/dInWH = dL/dW * pw * exp(in_w)
        //as w = pw * exp(in_w) and dW/din_w = w
        INDArray dLc_din_wh = dLc_dwh.muli(predictedWH);
        INDArray dLc_din_xy = new ActivationSigmoid().backprop(preSigmoidPredictedXYCenterGrid, dLc_dxy).getFirst();    //Shape: same as subset of input... [mb, b, 2, h, w]

        //Finally, apply masks: dLc_dwh and dLc_dxy should be 0 if no object is present in that box
        //Apply mask 1^obj_ij with shape [mb, b, h, w]
        Broadcast.mul(dLc_din_wh, mask1_ij_obj, dLc_din_wh, 0, 1, 3, 4);
        Broadcast.mul(dLc_din_xy, mask1_ij_obj, dLc_din_xy, 0, 1, 3, 4);


        epsWH.addi(dLc_din_wh);
        epsXY.addi(dLc_din_xy);

        return epsOut;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        return YoloUtils.activate(layerConf().getBoundingBoxes(), input, workspaceMgr);
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public boolean needsLabels() {
        return true;
    }

    @Override
    public double computeScore(double fullNetRegTerm, boolean training, LayerWorkspaceMgr workspaceMgr) {
        this.fullNetRegTerm = fullNetRegTerm;

        computeBackpropGradientAndScore(workspaceMgr, true, false);
        return score();
    }

    @Override
    public double score(){
        return score;
    }

    /**
     * Calculate IOU(truth, predicted) and gradients. Returns 5d arrays [mb, b, 2, H, W]
     * ***NOTE: All labels - and predicted values - are in terms of grid units - 0 to 12 usually, with default config ***
     *
     * @param labelTL   4d [mb, 2, H, W], label top/left (x,y) in terms of grid boxes
     * @param labelBR   4d [mb, 2, H, W], label bottom/right (x,y) in terms of grid boxes
     * @param predictedWH 5d [mb, b, 2, H, W] - predicted H/W in terms of number of grid boxes.
     * @param predictedXYinGridBox 5d [mb, b, 2, H, W] - predicted X/Y in terms of number of grid boxes. Values 0 to 1, center box value being 0.5
     * @param objectPresentMask 3d [mb, H, W] - mask array, for objects present (1) or not (0) in grid cell
     * @return IOU and gradients
     */
    private static IOURet calculateIOULabelPredicted(INDArray labelTL, INDArray labelBR, INDArray predictedWH, INDArray predictedXYinGridBox, INDArray objectPresentMask, INDArray objectPresentMaskBool){

        long mb = labelTL.size(0);
        long h = labelTL.size(2);
        long w = labelTL.size(3);
        long b = predictedWH.size(1);

        INDArray labelWH = labelBR.sub(labelTL);                //4d [mb, 2, H, W], label W/H in terms of number of grid boxes

        long gridH = labelTL.size(2);
        long gridW = labelTL.size(3);
        //Add grid positions to the predicted XY values (to get predicted XY in terms of grid cell units in image,
        // from (0 to 1 in grid cell) format)
        INDArray linspaceX = Nd4j.linspace(0, gridW-1, gridW, predictedWH.dataType());
        INDArray linspaceY = Nd4j.linspace(0, gridH-1, gridH, predictedWH.dataType());
        INDArray grid = Nd4j.createUninitialized(predictedWH.dataType(), new long[]{2, gridH, gridW}, 'c');
        INDArray gridX = grid.get(point(0), all(), all());
        INDArray gridY = grid.get(point(1), all(), all());
        Broadcast.copy(gridX, linspaceX, gridX, 1);
        Broadcast.copy(gridY, linspaceY, gridY, 0);

        //Calculate X/Y position overall (in grid box units) from "position in current grid box" format
        INDArray predictedXY = predictedXYinGridBox.ulike();;
        Broadcast.add(predictedXYinGridBox, grid, predictedXY, 2,3,4); // [2, H, W] to [mb, b, 2, H, W]


        INDArray halfWH = predictedWH.mul(0.5);
        INDArray predictedTL_XY = halfWH.rsub(predictedXY);     //xy - 0.5 * wh
        INDArray predictedBR_XY = halfWH.add(predictedXY);      //xy + 0.5 * wh

        INDArray maxTL = predictedTL_XY.ulike();   //Shape: [mb, b, 2, H, W]
        Broadcast.max(predictedTL_XY, labelTL, maxTL, 0, 2, 3, 4);
        INDArray minBR = predictedBR_XY.ulike();
        Broadcast.min(predictedBR_XY, labelBR, minBR, 0, 2, 3, 4);

        INDArray diff = minBR.sub(maxTL);
        INDArray intersectionArea = diff.prod(2);   //[mb, b, 2, H, W] to [mb, b, H, W]
        Broadcast.mul(intersectionArea, objectPresentMask, intersectionArea, 0, 2, 3);

        //Need to mask the calculated intersection values, to avoid returning non-zero values when intersection is actually 0
        //No intersection if: xP + wP/2 < xL - wL/2 i.e., BR_xPred < TL_xLab   OR  TL_xPred > BR_xLab (similar for Y axis)
        //Here, 1 if intersection exists, 0 otherwise. This is doing x/w and y/h simultaneously
        INDArray noIntMask1 = Nd4j.createUninitialized(DataType.BOOL, maxTL.shape(), maxTL.ordering());
        INDArray noIntMask2 = Nd4j.createUninitialized(DataType.BOOL, maxTL.shape(), maxTL.ordering());
        //Does both x and y on different dims
        Broadcast.lte(predictedBR_XY, labelTL, noIntMask1, 0, 2, 3, 4);  //Predicted BR <= label TL
        Broadcast.gte(predictedTL_XY, labelBR, noIntMask2, 0, 2, 3, 4);  //predicted TL >= label BR

        noIntMask1 = Transforms.or(noIntMask1.get(all(), all(), point(0), all(), all()), noIntMask1.get(all(), all(), point(1), all(), all()) );    //Shape: [mb, b, H, W]. Values 1 if no intersection
        noIntMask2 = Transforms.or(noIntMask2.get(all(), all(), point(0), all(), all()), noIntMask2.get(all(), all(), point(1), all(), all()) );
        INDArray noIntMask = Transforms.or(noIntMask1, noIntMask2 );

        INDArray intMask = Transforms.not(noIntMask); //Values 0 if no intersection
        Broadcast.mul(intMask, objectPresentMaskBool, intMask, 0, 2, 3);

        //Mask the intersection area: should be 0 if no intersection
        intMask = intMask.castTo(predictedWH.dataType());
        intersectionArea.muli(intMask);


        //Next, union area is simple: U = A1 + A2 - intersection
        INDArray areaPredicted = predictedWH.prod(2);   //[mb, b, 2, H, W] to [mb, b, H, W]
        Broadcast.mul(areaPredicted, objectPresentMask, areaPredicted, 0,2,3);
        INDArray areaLabel = labelWH.prod(1);           //[mb, 2, H, W] to [mb, H, W]

        INDArray unionArea = Broadcast.add(areaPredicted, areaLabel, areaPredicted.dup(), 0, 2, 3);
        unionArea.subi(intersectionArea);
        unionArea.muli(intMask);

        INDArray iou = intersectionArea.div(unionArea);
        BooleanIndexing.replaceWhere(iou, 0.0, Conditions.isNan()); //0/0 -> NaN -> 0

        //Apply the "object present" mask (of shape [mb, h, w]) - this ensures IOU is 0 if no object is present
        Broadcast.mul(iou, objectPresentMask, iou, 0, 2, 3);

        //Finally, calculate derivatives:
        INDArray maskMaxTL = Nd4j.createUninitialized(DataType.BOOL, maxTL.shape(), maxTL.ordering());    //1 if predicted Top/Left is max, 0 otherwise
        Broadcast.gt(predictedTL_XY, labelTL, maskMaxTL, 0, 2, 3, 4);   // z = x > y
        maskMaxTL = maskMaxTL.castTo(predictedWH.dataType());

        INDArray maskMinBR = Nd4j.createUninitialized(DataType.BOOL, maxTL.shape(), maxTL.ordering());    //1 if predicted Top/Left is max, 0 otherwise
        Broadcast.lt(predictedBR_XY, labelBR, maskMinBR, 0, 2, 3, 4);   // z = x < y
        maskMinBR = maskMinBR.castTo(predictedWH.dataType());

        //dI/dx = lambda * (1^(min(x1+w1/2) - 1^(max(x1-w1/2))
        //dI/dy = omega * (1^(min(y1+h1/2) - 1^(max(y1-h1/2))
        //omega = min(x1+w1/2,x2+w2/2) - max(x1-w1/2,x2+w2/2)       i.e., from diff = minBR.sub(maxTL), which has shape [mb, b, 2, h, w]
        //lambda = min(y1+h1/2,y2+h2/2) - max(y1-h1/2,y2+h2/2)
        INDArray dI_dxy = maskMinBR.sub(maskMaxTL);              //Shape: [mb, b, 2, h, w]
        INDArray dI_dwh = maskMinBR.addi(maskMaxTL).muli(0.5);    //Shape: [mb, b, 2, h, w]

        dI_dxy.get(all(), all(), point(0), all(), all()).muli(diff.get(all(), all(), point(1), all(), all()));
        dI_dxy.get(all(), all(), point(1), all(), all()).muli(diff.get(all(), all(), point(0), all(), all()));

        dI_dwh.get(all(), all(), point(0), all(), all()).muli(diff.get(all(), all(), point(1), all(), all()));
        dI_dwh.get(all(), all(), point(1), all(), all()).muli(diff.get(all(), all(), point(0), all(), all()));

        //And derivatives WRT IOU:
        INDArray uPlusI = unionArea.add(intersectionArea);
        INDArray u2 = unionArea.mul(unionArea);
        INDArray uPlusIDivU2 = uPlusI.div(u2);   //Shape: [mb, b, h, w]
        BooleanIndexing.replaceWhere(uPlusIDivU2, 0.0, Conditions.isNan());     //Handle 0/0

        INDArray dIOU_dxy = Nd4j.createUninitialized(predictedWH.dataType(), new long[]{mb, b, 2, h, w}, 'c');
        Broadcast.mul(dI_dxy, uPlusIDivU2, dIOU_dxy, 0, 1, 3, 4);   //[mb, b, h, w] x [mb, b, 2, h, w]

        INDArray predictedHW = Nd4j.createUninitialized(predictedWH.dataType(), new long[]{mb, b, 2, h, w}, predictedWH.ordering());
        //Next 2 lines: permuting the order... WH to HW along dimension 2
        predictedHW.get(all(), all(), point(0), all(), all()).assign(predictedWH.get(all(), all(), point(1), all(), all()));
        predictedHW.get(all(), all(), point(1), all(), all()).assign(predictedWH.get(all(), all(), point(0), all(), all()));

        INDArray Ihw = predictedHW.ulike();;
        Broadcast.mul(predictedHW, intersectionArea, Ihw, 0, 1, 3, 4 );    //Predicted_wh: [mb, b, 2, h, w]; intersection: [mb, b, h, w]

        INDArray dIOU_dwh = Nd4j.createUninitialized(predictedHW.dataType(), new long[]{mb, b, 2, h, w}, 'c');
        Broadcast.mul(dI_dwh, uPlusI, dIOU_dwh, 0, 1, 3, 4);
        dIOU_dwh.subi(Ihw);
        Broadcast.div(dIOU_dwh, u2, dIOU_dwh, 0, 1, 3, 4);
        BooleanIndexing.replaceWhere(dIOU_dwh, 0.0, Conditions.isNan());     //Handle division by 0 (due to masking, etc)

        return new IOURet(iou, dIOU_dxy, dIOU_dwh);
    }


    @AllArgsConstructor
    @Data
    private static class IOURet {
        private INDArray iou;
        private INDArray dIOU_dxy;
        private INDArray dIOU_dwh;

    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr){

        //TODO
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetRegTerm, LayerWorkspaceMgr workspaceMgr) {
        this.fullNetRegTerm = fullNetRegTerm;
        return computeBackpropGradientAndScore(workspaceMgr, false, true);
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

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    /** @see YoloUtils#getPredictedObjects(INDArray, INDArray, double, double) */
    public List<DetectedObject> getPredictedObjects(INDArray networkOutput, double threshold){
        return YoloUtils.getPredictedObjects(layerConf().getBoundingBoxes(), networkOutput, threshold, 0.0);
    }

    /**
     * Get the confidence matrix (confidence for all x/y positions) for the specified bounding box, from the network
     * output activations array
     *
     * @param networkOutput Network output activations
     * @param example       Example number, in minibatch
     * @param bbNumber      Bounding box number
     * @return Confidence matrix
     */
    public INDArray getConfidenceMatrix(INDArray networkOutput, int example, int bbNumber){

        //Input format: [minibatch, 5B+C, H, W], with order [x,y,w,h,c]
        //Therefore: confidences are at depths 4 + bbNumber * 5

        INDArray conf = networkOutput.get(point(example), point(4+bbNumber*5), all(), all());
        return conf;
    }

    /**
     * Get the probability matrix (probability of the specified class, assuming an object is present, for all x/y
     * positions), from the network output activations array
     *
     * @param networkOutput Network output activations
     * @param example       Example number, in minibatch
     * @param classNumber   Class number
     * @return Confidence matrix
     */
    public INDArray getProbabilityMatrix(INDArray networkOutput, int example, int classNumber){
        //Input format: [minibatch, 5B+C, H, W], with order [x,y,w,h,c]
        //Therefore: probabilities for class I is at depths 5B + classNumber

        val bbs = layerConf().getBoundingBoxes().size(0);
        INDArray conf = networkOutput.get(point(example), point(5*bbs + classNumber), all(), all());
        return conf;
    }
}
