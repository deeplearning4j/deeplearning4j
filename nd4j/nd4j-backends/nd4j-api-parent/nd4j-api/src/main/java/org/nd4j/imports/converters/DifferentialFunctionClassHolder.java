/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.converters;

import dorkbox.annotation.AnnotationDefaults;
import dorkbox.annotation.AnnotationDetector;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.onnx.OnnxDescriptorParser;
import org.nd4j.imports.descriptors.onnx.OpDescriptor;
import org.nd4j.imports.descriptors.tensorflow.TensorflowDescriptorParser;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.custom.Invoke;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.ops.impl.shape.SetShape;
import org.nd4j.linalg.api.ops.random.impl.CustomDropOut;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.OpDef;

import java.io.IOException;
import java.lang.annotation.ElementType;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
public class DifferentialFunctionClassHolder {
    private static Map<Long,Class<?>> customOpHashToClass = new HashMap<>();
    private static Map<Long,Map<String,Class<?>>> customOpHashToClasses = new ConcurrentHashMap<>(); //Only contains ops with 1 hash to multiple classes
    private  static Map<String,Class<?>> udfs = new HashMap<>();
    private static List<String> missingOps = new ArrayList<>();

    private static  Map<String, DifferentialFunction> OP_NAME_MAP;

    private static  List<Class<?>> fnClasses;


    private static Map<String,Map<String,Field>> fieldsForFunction;

    private static  Set<String>  fieldNamesOpsIgnore;


    private static DifferentialFunctionClassHolder INSTANCE;

    //When determining fields/properties, where should we terminate the search?
    //We don't want to include every single field from every single superclass
    private static  Set<Class> classesToIgnore;

    private static  Map<Class<?>,Set<String>> classFieldsToIgnore;

    private static AtomicBoolean initialized = new AtomicBoolean(false);




    public static void initInstance() throws IOException {
        log.trace("Initializing DifferentialClassHolder");
        if(initialized.get())
            return;
        classesToIgnore = new HashSet<>(Arrays.<Class>asList(
                Object.class
        ));
        classFieldsToIgnore = new ConcurrentHashMap<>();
        classFieldsToIgnore.put(BaseOp.class, new HashSet<>(Arrays.asList("x", "y", "z", "n", "numProcessed", "xVertexId", "yVertexId", "zVertexId", "extraArgz")));
        log.trace("Initialized class fields");
        log.trace("Initializing import class mapping");
        OP_NAME_MAP = new ConcurrentHashMap<>();
        log.trace("Creating fn classes");
        fnClasses = new ArrayList<>(Arrays.<Class<?>>asList(
                org.nd4j.linalg.api.ops.DynamicCustomOp.class,
                org.nd4j.linalg.api.ops.NoOp.class,
                Invoke.class,
                org.nd4j.linalg.api.ops.impl.updaters.SgdUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.RmsPropUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.NesterovsUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.NadamUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.AmsGradUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.AdamUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.AdaMaxUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.AdaGradUpdater.class,
                org.nd4j.linalg.api.ops.impl.updaters.AdaDeltaUpdater.class,
                org.nd4j.linalg.api.ops.custom.BarnesEdgeForces.class,
                org.nd4j.linalg.api.ops.custom.BarnesHutGains.class,
                org.nd4j.linalg.api.ops.custom.BarnesHutSymmetrize.class,
                org.nd4j.linalg.api.ops.custom.KnnMinDistance.class,
                org.nd4j.linalg.api.ops.custom.SpTreeCell.class,
                org.nd4j.linalg.api.ops.custom.Flatten.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BiasAddGrad.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMax.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMin.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastGradientArgs.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMax.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMin.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastRDivOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastRSubOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp.class,
                org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo.class,
                org.nd4j.linalg.api.ops.impl.shape.Create.class,
                org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastEqualTo.class,
                org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThan.class,
                org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThanOrEqual.class,
                org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastLessThan.class,
                org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastLessThanOrEqual.class,
                org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastNotEqual.class,
                org.nd4j.linalg.api.ops.impl.controlflow.Select.class,
                org.nd4j.linalg.api.ops.impl.controlflow.Where.class,
                org.nd4j.linalg.api.ops.impl.controlflow.WhereNumpy.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.While.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.Exit.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.LoopCond.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.NextIteration.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.StopGradient.class,
                org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch.class,
                org.nd4j.linalg.api.ops.impl.grid.FreeGridOp.class,
                org.nd4j.linalg.api.ops.impl.image.CropAndResize.class,
                org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches.class,
                org.nd4j.linalg.api.ops.impl.image.ImageResize.class,
                org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression.class,
                org.nd4j.linalg.api.ops.impl.image.NonMaxSuppressionV3.class,
                org.nd4j.linalg.api.ops.impl.image.NonMaxSuppressionWithOverlaps.class,
                org.nd4j.linalg.api.ops.impl.image.ResizeBilinear.class,
                org.nd4j.linalg.api.ops.impl.image.ResizeBicubic.class,
                org.nd4j.linalg.api.ops.impl.image.ResizeNearestNeighbor.class,
                SetShape.class,
                org.nd4j.linalg.api.ops.impl.image.ResizeArea.class,
                org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex.class,
                org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex.class,
                org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax.class,
                org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin.class,
                org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgAmax.class,
                org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgAmin.class,
                org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling3D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNormDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2DTF.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3DTF.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2DBp.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Im2colBp.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalization.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalizationDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling3D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPoolWithArgmax.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2DDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.SpaceToDepth.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling3d.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling3dBp.class,
                org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2dDerivative.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.GRU.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUBp.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMCell.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayerBp.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlock.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU.class,
                org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell.class,
                org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.HingeLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.HuberLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.L2Loss.class,
                org.nd4j.linalg.api.ops.impl.loss.LogLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyWithLogitsLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.SparseSoftmaxCrossEntropyLossWithLogits.class,
                org.nd4j.linalg.api.ops.impl.loss.WeightedCrossEntropyLoss.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.AbsoluteDifferenceLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.CosineDistanceLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.HingeLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.HuberLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.LogLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.LogPoissonLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.MeanPairwiseSquaredErrorLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.MeanSquaredErrorLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.SigmoidCrossEntropyLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.SoftmaxCrossEntropyLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.SoftmaxCrossEntropyWithLogitsLossBp.class,
                org.nd4j.linalg.api.ops.impl.loss.bp.SparseSoftmaxCrossEntropyLossWithLogitsBp.class,
                org.nd4j.linalg.api.ops.impl.meta.InvertedPredicateMetaOp.class,
                org.nd4j.linalg.api.ops.impl.meta.PostulateMetaOp.class,
                org.nd4j.linalg.api.ops.impl.meta.PredicateMetaOp.class,
                org.nd4j.linalg.api.ops.impl.meta.ReduceMetaOp.class,
                org.nd4j.linalg.api.ops.impl.nlp.CbowRound.class,
                org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound.class,
                org.nd4j.linalg.api.ops.impl.reduce.HashCode.class,
                org.nd4j.linalg.api.ops.impl.reduce.Mmul.class,
                org.nd4j.linalg.api.ops.impl.reduce.MmulBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.Moments.class,
                org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments.class,
                org.nd4j.linalg.api.ops.impl.reduce.SufficientStatistics.class,
                org.nd4j.linalg.api.ops.impl.reduce.TensorMmul.class,
                org.nd4j.linalg.api.ops.impl.reduce.ZeroFraction.class,
                org.nd4j.linalg.api.ops.impl.reduce.bool.All.class,
                org.nd4j.linalg.api.ops.impl.reduce.bool.Any.class,
                org.nd4j.linalg.api.ops.impl.reduce.bool.IsInf.class,
                org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.CumProdBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.CumSumBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.DotBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.MaxBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.MeanBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.MinBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.Norm1Bp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.Norm2Bp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.NormMaxBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.ProdBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.SquaredNormBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.StandardDeviationBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.VarianceBp.class,
                org.nd4j.linalg.api.ops.impl.reduce.custom.BatchMmul.class,
                org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.AMean.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.Mean.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy.class,
                org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm.class,
                org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero.class,
                org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero.class,
                org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.AMax.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.AMin.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.ASum.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.Max.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.Min.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.Prod.class,
                org.nd4j.linalg.api.ops.impl.reduce.same.Sum.class,
                org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance.class,
                org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity.class,
                org.nd4j.linalg.api.ops.impl.reduce3.Dot.class,
                org.nd4j.linalg.api.ops.impl.reduce3.EqualsWithEps.class,
                org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance.class,
                org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance.class,
                org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance.class,
                org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance.class,
                org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU.class,
                org.nd4j.linalg.api.ops.impl.scalar.LogX.class,
                org.nd4j.linalg.api.ops.impl.scalar.Pow.class,
                org.nd4j.linalg.api.ops.impl.scalar.PowDerivative.class,
                org.nd4j.linalg.api.ops.impl.reduce.bp.PowBp.class,
                org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear.class,
                org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinearDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ThresholdRelu.class,
                org.nd4j.linalg.api.ops.impl.scalar.Relu6.class,
                org.nd4j.linalg.api.ops.impl.scalar.PRelu.class,
                org.nd4j.linalg.api.ops.impl.scalar.ReplaceNans.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarMax.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarMin.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarRemainder.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseDivision.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseSubtraction.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarSet.class,
                org.nd4j.linalg.api.ops.impl.scalar.ScalarSubtraction.class,
                org.nd4j.linalg.api.ops.impl.scalar.Step.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarAnd.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEps.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThanOrEqual.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThanOrEqual.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNot.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarOr.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue.class,
                org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarXor.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterAdd.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterDiv.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterMax.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterMin.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterMul.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterNd.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterNdAdd.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterNdSub.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterNdUpdate.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterSub.class,
                org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate.class,
                org.nd4j.linalg.api.ops.impl.shape.ApplyGradientDescent.class,
                org.nd4j.linalg.api.ops.impl.shape.BroadcastDynamicShape.class,
                org.nd4j.linalg.api.ops.impl.shape.Concat.class,
                org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix.class,
                org.nd4j.linalg.api.ops.impl.shape.Cross.class,
                org.nd4j.linalg.api.ops.impl.shape.Diag.class,
                org.nd4j.linalg.api.ops.impl.shape.DiagPart.class,
                org.nd4j.linalg.api.ops.impl.shape.ExpandDims.class,
                org.nd4j.linalg.api.ops.impl.shape.Eye.class,
                org.nd4j.linalg.api.ops.impl.shape.Flatten2D.class,
                org.nd4j.linalg.api.ops.impl.shape.Gather.class,
                org.nd4j.linalg.api.ops.impl.shape.GatherNd.class,
                org.nd4j.linalg.api.ops.impl.shape.Linspace.class,
                org.nd4j.linalg.api.ops.impl.shape.MergeAvg.class,
                org.nd4j.linalg.api.ops.impl.shape.MergeMax.class,
                org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex.class,
                org.nd4j.linalg.api.ops.impl.shape.MergeSum.class,
                org.nd4j.linalg.api.ops.impl.shape.MeshGrid.class,
                org.nd4j.linalg.api.ops.impl.shape.OneHot.class,
                org.nd4j.linalg.api.ops.impl.shape.OnesLike.class,
                org.nd4j.linalg.api.ops.impl.shape.ParallelStack.class,
                org.nd4j.linalg.api.ops.impl.shape.Permute.class,
                org.nd4j.linalg.api.ops.impl.shape.Rank.class,
                org.nd4j.linalg.api.ops.impl.shape.ReductionShape.class,
                org.nd4j.linalg.api.ops.impl.shape.Repeat.class,
                org.nd4j.linalg.api.ops.impl.shape.Reshape.class,
                org.nd4j.linalg.api.ops.impl.shape.SequenceMask.class,
                org.nd4j.linalg.api.ops.impl.shape.Shape.class,
                org.nd4j.linalg.api.ops.impl.shape.ShapeN.class,
                org.nd4j.linalg.api.ops.impl.shape.Size.class,
                org.nd4j.linalg.api.ops.impl.shape.SizeAt.class,
                org.nd4j.linalg.api.ops.impl.shape.Slice.class,
                org.nd4j.linalg.api.ops.impl.shape.Split.class,
                org.nd4j.linalg.api.ops.impl.shape.SplitV.class,
                org.nd4j.linalg.api.ops.impl.shape.Squeeze.class,
                org.nd4j.linalg.api.ops.impl.shape.Stack.class,
                org.nd4j.linalg.api.ops.impl.shape.StridedSlice.class,
                org.nd4j.linalg.api.ops.impl.shape.Tile.class,
                org.nd4j.linalg.api.ops.impl.shape.Transpose.class,
                org.nd4j.linalg.api.ops.impl.shape.Unstack.class,
                org.nd4j.linalg.api.ops.impl.shape.ZerosLike.class,
                org.nd4j.linalg.api.ops.impl.shape.bp.ConcatBp.class,
                org.nd4j.linalg.api.ops.impl.shape.bp.MergeMaxBp.class,
                org.nd4j.linalg.api.ops.impl.shape.bp.MergeAvgBp.class,
                org.nd4j.linalg.api.ops.impl.shape.bp.SliceBp.class,
                org.nd4j.linalg.api.ops.impl.shape.bp.StridedSliceBp.class,
                org.nd4j.linalg.api.ops.impl.shape.bp.TileBp.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.EmbeddingLookup.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayConcat.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayGather.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayRead.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayScatter.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArraySize.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArraySplit.class,
                org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayWrite.class,
                org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation.class,
                org.nd4j.linalg.api.ops.impl.summarystats.Variance.class,
                org.nd4j.linalg.api.ops.impl.transforms.Angle.class,
                org.nd4j.linalg.api.ops.impl.transforms.Assert.class,
                org.nd4j.linalg.api.ops.impl.transforms.BinCount.class,
                org.nd4j.linalg.api.ops.impl.transforms.CheckNumerics.class,
                org.nd4j.linalg.api.ops.impl.transforms.Cholesky.class,
                org.nd4j.linalg.api.ops.impl.transforms.Histogram.class,
                org.nd4j.linalg.api.ops.impl.transforms.HistogramFixedWidth.class,
                org.nd4j.linalg.api.ops.impl.transforms.IdentityN.class,
                org.nd4j.linalg.api.ops.impl.transforms.MaxOut.class,
                org.nd4j.linalg.api.ops.impl.transforms.NthElement.class,
                org.nd4j.linalg.api.ops.impl.transforms.Pad.class,
                org.nd4j.linalg.api.ops.impl.transforms.ReluLayer.class,
                org.nd4j.linalg.api.ops.impl.transforms.any.Assign.class,
                org.nd4j.linalg.api.ops.impl.transforms.any.IsMax.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.BooleanNot.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform.class,
                org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm.class,
                org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm.class,
                org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNormBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue.class,
                org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace.class,
                org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet.class,
                org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ATan2.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Assign.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpace.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpaceND.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Choose.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.CReLU.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.CReluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.BitsHammingDistance.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseAnd.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseXor.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseOr.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Dilation2D.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttention.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2Bp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicPartition.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicStitch.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.EqualTo.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.FakeQuantWithMinMaxArgs.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.FakeQuantWithMinMaxVars.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Fill.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThan.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThanOrEqual.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.InTopK.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.InvertPermutation.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.IsNumericTensor.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNormBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LessThan.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LessThanOrEqual.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ListDiff.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LogMatrixDeterminant.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalAnd.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalNot.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalOr.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalXor.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDeterminant.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDiag.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDiagPart.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixInverse.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixSetDiag.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Max.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MaximumBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Min.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MirrorPad.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttentionBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.NotEqualTo.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ParallelConcat.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Pow.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Reverse.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseV2.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.SpaceToBatch.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.SpaceToBatchND.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.StandardizeBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Svd.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.TopK.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Trace.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Unique.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.UniqueWithCounts.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Zeta.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMax.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMean.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMin.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentProd.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSum.class,
                org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast.class,
                org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt.class,
                org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.DynamicPartitionBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.LogSoftMaxDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.RectifiedTanhDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.Relu6Derivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.PReluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SELUDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.EluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.RectifiedTanhBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SeluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftPlusBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.ThresholdReluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryMinimalRelativeError.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryRelativeError.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.RelativeError.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.Set.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.Axpy.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.CopyOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FModOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorDivOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorModOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MergeAddOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.ModOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.PowPairwise.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RDivOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RSubOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RealDivOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RemainderOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SquaredDifferenceOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.TruncateDivOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.AddBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.DivBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.FloorDivBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.FloorModBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.ModBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.MergeAddBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.MulBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.RDivBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.RSubBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.SquaredDifferenceBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.SubBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.SubBpOp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Not.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.AMax.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.AMin.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Abs.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Ceil.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Cube.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Floor.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Identity.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Max.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Min.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Negative.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.OneMinus.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Round.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Sign.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.Square.class,
                org.nd4j.linalg.api.ops.impl.transforms.same.TimesOneMinus.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMin.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentProd.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSqrtN.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentMaxBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentMeanBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentMinBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentProdBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentSumBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentMaxBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentMeanBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentMinBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentProdBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentSqrtNBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentSumBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ACos.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ASin.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ATan.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ATanh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Cos.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Cosh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.ELU.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Erf.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Exp.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.GELU.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.GELUDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.HardTanh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Log.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.LogSigmoid.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Mish.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.MishDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELU.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELUDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.RationalTanh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.RectifiedTanh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Rint.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SELU.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SetRange.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Sin.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Sinh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SoftSign.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Stabilize.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Swish.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SwishDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Tan.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.TanDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative.class,
                org.nd4j.linalg.api.ops.persistence.RestoreV2.class,
                org.nd4j.linalg.api.ops.persistence.SaveV2.class,
                org.nd4j.linalg.api.ops.random.impl.RandomMultinomial.class,
                org.nd4j.linalg.api.ops.random.compat.RandomStandardNormal.class,
                org.nd4j.linalg.api.ops.random.custom.DistributionUniform.class,
                org.nd4j.linalg.api.ops.random.custom.RandomBernoulli.class,
                org.nd4j.linalg.api.ops.random.custom.RandomExponential.class,
                org.nd4j.linalg.api.ops.random.custom.RandomNormal.class,
                org.nd4j.linalg.api.ops.random.custom.RandomGamma.class,
                org.nd4j.linalg.api.ops.random.custom.RandomPoisson.class,
                org.nd4j.linalg.api.ops.random.custom.RandomShuffle.class,
                org.nd4j.linalg.api.ops.random.impl.AlphaDropOut.class,
                CustomDropOut.class,
                org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution.class,
                org.nd4j.linalg.api.ops.random.impl.BinomialDistribution.class,
                org.nd4j.linalg.api.ops.random.impl.BinomialDistributionEx.class,
                org.nd4j.linalg.api.ops.random.impl.Choice.class,
                org.nd4j.linalg.api.ops.random.impl.DropOutInverted.class,
                org.nd4j.linalg.api.ops.random.impl.GaussianDistribution.class,
                org.nd4j.linalg.api.ops.random.impl.Linspace.class,
                org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution.class,
                org.nd4j.linalg.api.ops.random.impl.ProbablisticMerge.class,
                org.nd4j.linalg.api.ops.random.impl.Range.class,
                org.nd4j.linalg.api.ops.random.impl.TruncatedNormalDistribution.class,
                org.nd4j.linalg.api.ops.random.impl.UniformDistribution.class,
                org.nd4j.linalg.api.ops.util.PrintAffinity.class,
                org.nd4j.linalg.api.ops.util.PrintVariable.class,
                org.nd4j.linalg.api.ops.compat.CompatSparseToDense.class,
                org.nd4j.linalg.api.ops.compat.CompatStringSplit.class,
                org.nd4j.linalg.api.ops.custom.AdjustContrast.class,
                org.nd4j.linalg.api.ops.custom.HsvToRgb.class,
                org.nd4j.linalg.api.ops.custom.RgbToHsv.class,
                org.nd4j.linalg.api.ops.custom.RgbToYiq.class,
                org.nd4j.linalg.api.ops.custom.RgbToGrayscale.class,
                org.nd4j.linalg.api.ops.custom.YiqToRgb.class,
                org.nd4j.linalg.api.ops.custom.RgbToYuv.class,
                org.nd4j.linalg.api.ops.custom.YuvToRgb.class,
                org.nd4j.linalg.api.ops.custom.BitCast.class,
                org.nd4j.linalg.api.ops.custom.CompareAndBitpack.class,
                org.nd4j.linalg.api.ops.custom.DivideNoNan.class,
                org.nd4j.linalg.api.ops.custom.DrawBoundingBoxes.class,
                org.nd4j.linalg.api.ops.custom.FakeQuantWithMinMaxVarsPerChannel.class,
                org.nd4j.linalg.api.ops.custom.AdjustSaturation.class,
                org.nd4j.linalg.api.ops.custom.AdjustHue.class,
                org.nd4j.linalg.api.ops.custom.FusedBatchNorm.class,
                org.nd4j.linalg.api.ops.custom.BetaInc.class,
                org.nd4j.linalg.api.ops.custom.MatrixBandPart.class,
                org.nd4j.linalg.api.ops.custom.Polygamma.class,
                org.nd4j.linalg.api.ops.custom.Lgamma.class,
                org.nd4j.linalg.api.ops.custom.RandomCrop.class,
                org.nd4j.linalg.api.ops.custom.Roll.class,
                org.nd4j.linalg.api.ops.custom.ToggleBits.class,
                org.nd4j.linalg.api.ops.custom.Tri.class,
                org.nd4j.linalg.api.ops.custom.Triu.class,
                org.nd4j.linalg.api.ops.custom.TriuBp.class,
                org.nd4j.linalg.api.ops.custom.Igamma.class,
                org.nd4j.linalg.api.ops.custom.Igammac.class,
                org.nd4j.linalg.api.ops.custom.Digamma.class,
                org.nd4j.linalg.api.ops.custom.Lu.class,
                org.nd4j.linalg.api.ops.custom.TriangularSolve.class,
                org.nd4j.linalg.api.ops.custom.LinearSolve.class,
                org.nd4j.linalg.api.ops.custom.Lstsq.class,
                org.nd4j.linalg.api.ops.impl.transforms.custom.Qr.class,
                org.nd4j.linalg.api.ops.custom.Logdet.class
        ));

        System.out.println("Created fn classes");
        // Get a list of all classes annotated with @UserDefinedOp,
        if(System.getProperties().containsKey(ND4JSystemProperties.UDF_NAME_SPACES)) {
            log.trace("In udf namespaces with scanning");
            String[] packageNames = System.getProperty(ND4JSystemProperties.UDF_NAME_SPACES).split(",");
            log.trace("Package names " + Arrays.toString(packageNames));
            ClassLoader nd4jClassloader = ND4JClassLoading.getNd4jClassloader();
            log.trace("Nd4j class loader " + nd4jClassloader);
            List<Class<?>> classModules = AnnotationDetector.scanClassPath(nd4jClassloader,packageNames)
                    .forAnnotations(UserDefinedOp.class)  // one or more annotations
                    .on(ElementType.TYPE) // optional, default ElementType.TYPE. One ore more element types
                    .collect(AnnotationDefaults.getType);
            log.trace("Class modules " + classModules);
            classModules.forEach(udf -> fnClasses.add(udf));
            log.trace("Done with scanning");
        }



        System.out.println("Populating op map");
        OP_NAME_MAP = new ConcurrentHashMap<>();
        for(Class<?> c : fnClasses) {
            try {
                System.out.println("Initializing " + c.getName());
                DifferentialFunction df = (DifferentialFunction) c.newInstance();
                if(df == null)
                    continue;
                String opName = df.opName();
                if(opName != null)
                    OP_NAME_MAP.put(opName, df);

            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }

        System.out.println("Populated op map");


        fieldNamesOpsIgnore = new LinkedHashSet<>() {{
            add("extraArgs");
            add("arrayInitialized");
            add("log");
            add("inputArguments");
            add("outputArguments");
            add("outputShapes");
            add("outputVariables");
            add("tArguments");
            add("iArguments");
            add("bArguments");
            add("dArguments");
            add("hash");
            add("opName");
            add("sameDiff");
            add("ownName");
        }};
        System.out.println("Initialized field names ops ignore");


        fieldsForFunction = new LinkedHashMap<>();
        for(DifferentialFunction df : OP_NAME_MAP.values()) {
            if(df == null || df.opName() == null) {
                continue;
            }
            try {
                //accumulate the field names for a given function
                //this is mainly used in import
                Map<String, Field> fieldNames = new LinkedHashMap<>();
                Class<? extends DifferentialFunction> current = df.getClass();
                System.out.println("Setting up fields for function processing: " + current.getName());
                val fields = new ArrayList<Field>();
                boolean isFirst = true;

                while (current.getSuperclass() != null && !classesToIgnore.contains(current.getSuperclass())) {

                    if (df.isConfigProperties() && isFirst) {

                        String fieldName = df.configFieldName();

                        if(fieldName == null)
                            fieldName = "config";

                        Field configField = null;
                        try{
                            configField = current.getDeclaredField(fieldName);
                        } catch (NoSuchFieldException e){
                            Class<?> currentConfig = current.getSuperclass();

                            // find a config field in superclasses
                            while(currentConfig.getSuperclass() != null) {
                                try {
                                    configField = currentConfig.getDeclaredField(fieldName);
                                    break;
                                } catch (NoSuchFieldException e2) {
                                    currentConfig = currentConfig.getSuperclass();
                                }
                            }
                        }

                        if(configField == null)
                            continue;

                        val configFieldClass = configField.getType();

                        for (val field : configFieldClass.getDeclaredFields()) {
                            if (!Modifier.isStatic(field.getModifiers()) && !fieldNamesOpsIgnore.contains(field.getName()) &&
                                    (!classFieldsToIgnore.containsKey(current) || !classFieldsToIgnore.get(current).contains(field.getName()))) {
                                fields.add(field);
                                field.setAccessible(true);
                                if (fieldNames.containsKey(field.getName())) {
                                    throw new IllegalStateException("Field with name " + field.getName() + " exists for multiple classes: "
                                            + fieldNames.get(field.getName()).getDeclaringClass().getName() + " and " + field.getDeclaringClass().getName());
                                }
                                fieldNames.put(field.getName(), field);
                            }
                        }
                    } else {
                        for (Field field : current.getDeclaredFields()) {
                            if (!Modifier.isStatic(field.getModifiers()) && !fieldNamesOpsIgnore.contains(field.getName()) &&
                                    (!classFieldsToIgnore.containsKey(current) || !classFieldsToIgnore.get(current).contains(field.getName()))) {
                                fields.add(field);
                                field.setAccessible(true);
                                if (fieldNames.containsKey(field.getName())) {
                                    throw new IllegalStateException("Field with name " + field.getName() + " exists for multiple classes: "
                                            + fieldNames.get(field.getName()).getDeclaringClass().getName() + " and " + field.getDeclaringClass().getName());
                                }
                                fieldNames.put(field.getName(), field);
                            }
                        }
                    }

                    // do something with current's fields
                    current = (Class<? extends DifferentialFunction>) current.getSuperclass();
                    isFirst = false;

                }

                fieldsForFunction.put(df.getClass().getName(), fieldNames);
            } catch (NoOpNameFoundException e) {
               System.out.println("Skipping function  " + df.getClass());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }


        val map = new HashMap<>(Nd4j.getExecutioner().getCustomOperations());
        val set = map.keySet();
        set.removeAll(OP_NAME_MAP.keySet());
        missingOps.addAll(set);
        Collections.sort(missingOps);


        //Get custom ops - map from hash to class
        Map<String,CustomOpDescriptor> descriptorMap = Nd4j.getExecutioner().getCustomOperations();
        Set<Long> multiClassHashes = new HashSet<>();
        for (Map.Entry<String, CustomOpDescriptor> e : descriptorMap.entrySet()) {
            String name = e.getKey();
            DifferentialFunction df = getInstance(name);

            if (df == null) {
                //Can be no class for 2 reasons:
                //(a) op name aliases
                //(b) libnd4j ops with no corresponding ND4J op class
                continue;
            }

            if (!CustomOp.class.isAssignableFrom(df.getClass())) {
                //Not a custom op class
                continue;
            }

            long h = e.getValue().getHash();
            if (customOpHashToClass.containsKey(h)) {
                //One op hash mapped to multiple classes
                multiClassHashes.add(h);
            }
            customOpHashToClass.put(e.getValue().getHash(), df.getClass());
        }

        for (Map.Entry<String, CustomOpDescriptor> e : descriptorMap.entrySet()) {
            long h = e.getValue().getHash();
            if (multiClassHashes.contains(h)) {
                if (!customOpHashToClasses.containsKey(h)) {
                    customOpHashToClasses.put(h, new HashMap<>());
                }
                Map<String, Class<?>> m = customOpHashToClasses.get(h);
                String name = e.getKey();
                DifferentialFunction df = getInstance(name);
                if(df == null)
                    continue;
                m.put(e.getKey(), df.getClass());
            }
        }



        try {
            if(System.getProperties().containsKey(ND4JSystemProperties.UDF_CLASSES)) {
                String[] classNames = System.getProperty(ND4JSystemProperties.UDF_CLASSES).split(",");
                for(String className : classNames) {
                    Class<?> clazz = null;
                    try {
                        clazz = Class.forName(className);
                        UserDefinedCustomOp o = (UserDefinedCustomOp) clazz.newInstance();
                        udfs.put(o.opName(),clazz);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }

                }
            }

            // Get a list of all classes annotated with @UserDefinedOp,
            else  if(System.getProperties().containsKey(ND4JSystemProperties.UDF_NAME_SPACES)) {
                String[] packageNames = System.getProperty(ND4JSystemProperties.UDF_NAME_SPACES).split(",");
                List<Class<?>> classModules = AnnotationDetector.scanClassPath(ND4JClassLoading.getNd4jClassloader(),packageNames)
                        .forAnnotations(UserDefinedOp.class)  // one or more annotations
                        .on(ElementType.TYPE) // optional, default ElementType.TYPE. One ore more element types
                        .collect(AnnotationDefaults.getType);
                classModules.forEach(udf ->  {
                    try {
                        UserDefinedCustomOp o = (UserDefinedCustomOp) udf.newInstance();
                        udfs.put(o.opName(),udf);
                    } catch (InstantiationException e) {
                        throw new RuntimeException(e);
                    } catch (IllegalAccessException e) {
                        throw new RuntimeException(e);
                    }
                });
            }

        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to start the client", e);
        }



        INSTANCE = new DifferentialFunctionClassHolder();
        System.out.println("Initialized instance");

        initialized.set(true);
    }

    /**
     * Get the fields for a given {@link DifferentialFunction}
     * @param function the function to get the fields for
     * @return the fields for a given function
     */
    public Map<String,Field> getFieldsForFunction(DifferentialFunction function) {
        if(!fieldsForFunction.containsKey(function.getClass().getName())) {
            return Collections.emptyMap();
        }
        return fieldsForFunction.get(function.getClass().getName());
    }




    private DifferentialFunctionClassHolder() {

    }




    /**
     *
     * @param name
     * @return
     */
    public boolean hasName(String name) {
        return OP_NAME_MAP.containsKey(name);
    }


    public Set<String> opNames() {
        return OP_NAME_MAP.keySet();
    }

    /**
     *
     * @param name
     * @return
     */
    public static DifferentialFunction getInstance(String name) {
        return OP_NAME_MAP.get(name);
    }

    public Class<?> customOpClassForHashAndName(long customOpHash, String name) {
        log.trace("Finding custom op class name");
        switch (name) {
            case CreateView.OP_NAME:
                return CreateView.class;
            case Enter.OP_NAME:
                return Enter.class;
            case Exit.OP_NAME:
                return Exit.class;
            case NextIteration.OP_NAME:
                return NextIteration.class;
            case Merge.OP_NAME:
                return Merge.class;
            case Switch.OP_NAME:
                return Switch.class;
            case LoopCond.OP_NAME:
                return LoopCond.class;
            case ExternalErrorsFunction.OP_NAME:
                return ExternalErrorsFunction.class;
            default:
                if(udfs.containsKey(name)) {
                    return udfs.get(name);
                }
                if(customOpHashToClasses.containsKey(customOpHash)) {
                    return customOpHashToClasses.get(customOpHash).get(name);
                } else if(customOpHashToClass.containsKey(customOpHash)) {
                    return customOpHashToClass.get(customOpHash);
                } else if(OP_NAME_MAP.containsKey(name)) {
                    return OP_NAME_MAP.get(name).getClass();
                } else {
                    throw new IllegalStateException("No op known for hash: " + customOpHash + " and name " + name);
                }
        }

    }

    public static synchronized DifferentialFunctionClassHolder getInstance() {
        log.trace("Returning class holder instance");
        return INSTANCE;
    }


}
