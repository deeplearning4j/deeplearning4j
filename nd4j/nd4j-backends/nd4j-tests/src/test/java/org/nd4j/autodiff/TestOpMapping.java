/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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

package org.nd4j.autodiff;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.ImportClassMapping;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.NoOp;
import org.nd4j.linalg.api.ops.compat.CompatSparseToDense;
import org.nd4j.linalg.api.ops.compat.CompatStringSplit;
import org.nd4j.linalg.api.ops.custom.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.*;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.grid.FreeGridOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2DTF;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3DTF;
import org.nd4j.linalg.api.ops.impl.meta.InvertedPredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PostulateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.ReduceMetaOp;
import org.nd4j.linalg.api.ops.impl.nlp.CbowRound;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound;
import org.nd4j.linalg.api.ops.impl.reduce.HashCode;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue;
import org.nd4j.linalg.api.ops.impl.shape.ApplyGradientDescent;
import org.nd4j.linalg.api.ops.impl.shape.Create;
import org.nd4j.linalg.api.ops.impl.shape.ParallelStack;
import org.nd4j.linalg.api.ops.impl.transforms.any.Assign;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ParallelConcat;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryMinimalRelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryRelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.CopyOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.PowPairwise;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RealDivOp;
import org.nd4j.linalg.api.ops.impl.updaters.*;
import org.nd4j.linalg.api.ops.persistence.RestoreV2;
import org.nd4j.linalg.api.ops.persistence.SaveV2;
import org.nd4j.linalg.api.ops.util.PrintAffinity;
import org.nd4j.linalg.api.ops.util.PrintVariable;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.reflections.Reflections;

import java.io.File;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestOpMapping extends BaseNd4jTest {

    Set<Class<? extends DifferentialFunction>> subTypes;

    public TestOpMapping(Nd4jBackend b){
        super(b);

        Reflections reflections = new Reflections("org.nd4j");
        subTypes = reflections.getSubTypesOf(DifferentialFunction.class);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }



    @Test
    public void testOpMappingCoverage() throws Exception {
        Map<String, DifferentialFunction> opNameMapping = ImportClassMapping.getOpNameMapping();
        Map<String, DifferentialFunction> tfOpNameMapping = ImportClassMapping.getTFOpMappingFunctions();
        Map<String, DifferentialFunction> onnxOpNameMapping = ImportClassMapping.getOnnxOpMappingFunctions();


        for(Class<? extends DifferentialFunction> c : subTypes){

            if(Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers()) || ILossFunction.class.isAssignableFrom(c))
                continue;

            DifferentialFunction df;
            try {
                df = c.newInstance();
            } catch (Throwable t){
                //All differential functions should have a no-arg constructor
                throw new RuntimeException("Error instantiating new instance - op class " + c.getName() + " likely does not have a no-arg constructor", t);
            }
            String opName = df.opName();

            assertTrue("Op is missing - not defined in ImportClassMapping: " + opName +
                    "\nInstructions to fix: Add class to org.nd4j.imports.converters.ImportClassMapping", opNameMapping.containsKey(opName)
            );

            try{
                String[] tfNames = df.tensorflowNames();

                for(String s : tfNames ){
                    assertTrue("Tensorflow mapping not found: " + s, tfOpNameMapping.containsKey(s));
                    assertEquals("Tensorflow mapping: " + s, df.getClass(), tfOpNameMapping.get(s).getClass());
                }
            } catch (NoOpNameFoundException e){
                //OK, skip
            }


            try{
                String[] onnxNames = df.onnxNames();

                for(String s : onnxNames ){
                    assertTrue("Onnx mapping not found: " + s, onnxOpNameMapping.containsKey(s));
                    assertEquals("Onnx mapping: " + s, df.getClass(), onnxOpNameMapping.get(s).getClass());
                }
            } catch (NoOpNameFoundException e){
                //OK, skip
            }
        }
    }

    @Test
    public void testOpsInNamespace() throws Exception {
        //Ensure that every op is either in a namespace, OR it's explicitly marked as ignored (i.e., an op that we don't
        // want to add to a namespace for some reason)
        //Note that we ignore "*Bp", "*Gradient", "*Derivative" etc ops

        String path = FilenameUtils.concat(new File("").getAbsolutePath(), "../nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/ops");
        path = FilenameUtils.normalize(path);
        System.out.println(path);
        File dir = new File(path);
        Collection<File> c = FileUtils.listFiles(dir, new String[]{"java"}, true);

        String strPattern = " org.nd4j.linalg.api.ops(\\.(\\w)+)+";
        Pattern pattern = Pattern.compile(strPattern);


        Set<String> seenClasses = new HashSet<>();
        for(File f1 : c){
            List<String> lines = FileUtils.readLines(f1, StandardCharsets.UTF_8);
            for(String l : lines){
                Matcher matcher = pattern.matcher(l);
                while(matcher.find()){
                    int s = matcher.start();
                    int e = matcher.end();

                    String str = l.substring(s+1,e);        //+1 because pattern starts with space
                    seenClasses.add(str);
                }
            }
        }

        int countNotSeen = 0;
        int countSeen = 0;
        List<String> notSeen = new ArrayList<>();
        for(Class<? extends DifferentialFunction> cl : subTypes){
            String s = cl.getName();

            //Backprop/gradient ops should not be in namespaces
            if(s.endsWith("Bp") || s.endsWith("BpOp") || s.endsWith("Gradient") || s.endsWith("Derivative") || s.endsWith("Grad"))
                continue;

            if(Modifier.isAbstract(cl.getModifiers()) || Modifier.isInterface(cl.getModifiers()))       //Skip interfaces and abstract methods
                continue;

            if(excludedFromNamespaces.contains(cl))     //Explicitly excluded - don't want in namespaces
                continue;

            if(!seenClasses.contains(s)){
//                System.out.println("NOT SEEN: " + s);
                notSeen.add(s);
                countNotSeen++;
            } else {
                countSeen++;
            }
        }

        Collections.sort(notSeen);
        System.out.println(String.join("\n", notSeen));

        System.out.println("Not seen ops: " + countNotSeen);
        System.out.println("Seen ops: " + countSeen);
        System.out.println("Namespace scan count ops: " + seenClasses.size());
    }

    //Set of classes that we explicitly don't want in a namespace for some reason
    private static final Set<Class<? extends DifferentialFunction>> excludedFromNamespaces = new HashSet<>();
    static {
        Set<Class<? extends DifferentialFunction>> s = excludedFromNamespaces;

        //Updaters - used via TrainingConfig, not namespaces
        s.add(AdaDeltaUpdater.class);
        s.add(AdaGradUpdater.class);
        s.add(AdaMaxUpdater.class);
        s.add(AdamUpdater.class);
        s.add(AmsGradUpdater.class);
        s.add(NadamUpdater.class);
        s.add(NesterovsUpdater.class);
        s.add(RmsPropUpdater.class);
        s.add(SgdUpdater.class);

        //Legacy broadcast ops
        s.add(BroadcastAddOp.class);
        s.add(BroadcastAMax.class);
        s.add(BroadcastAMin.class);
        s.add(BroadcastCopyOp.class);
        s.add(BroadcastDivOp.class);
        s.add(BroadcastGradientArgs.class);
        s.add(BroadcastMax.class);
        s.add(BroadcastMin.class);
        s.add(BroadcastMulOp.class);
        s.add(BroadcastRDivOp.class);
        s.add(BroadcastRSubOp.class);
        s.add(BroadcastSubOp.class);
        s.add(BroadcastTo.class);
        s.add(BroadcastEqualTo.class);
        s.add(BroadcastGreaterThan.class);
        s.add(BroadcastGreaterThanOrEqual.class);
        s.add(BroadcastLessThan.class);
        s.add(BroadcastLessThanOrEqual.class);
        s.add(BroadcastNotEqual.class);

        //Redundant operations
        s.add(ArgMax.class);            //IMax already in namespace
        s.add(ArgMin.class);            //IMin already in namespace

        //Various utility methods, used internally
        s.add(DynamicCustomOp.class);
        s.add(ExternalErrorsFunction.class);
        s.add(GradientBackwardsMarker.class);
        s.add(KnnMinDistance.class);
        s.add(BinaryRelativeError.class);
        s.add(SpTreeCell.class);
        s.add(BarnesHutGains.class);
        s.add(BinaryMinimalRelativeError.class);
        s.add(SkipGramRound.class);
        s.add(BarnesHutSymmetrize.class);
        s.add(BarnesEdgeForces.class);
        s.add(CbowRound.class);

        //For TF compatibility only
        s.add(NoOp.class);
        s.add(RestoreV2.class);
        s.add(ParallelConcat.class);
        s.add(ParallelStack.class);
        s.add(DeConv2DTF.class);
        s.add(DeConv3DTF.class);
        s.add(CompatSparseToDense.class);
        s.add(CompatStringSplit.class);
        s.add(ApplyGradientDescent.class);
        s.add(RealDivOp.class);
        s.add(SaveV2.class);

        //Control ops, used internally as part of loops etc
        s.add(Enter.class);
        s.add(Exit.class);
        s.add(NextIteration.class);
        s.add(LoopCond.class);
        s.add(Merge.class);
        s.add(Switch.class);

        //MetaOps, grid ops etc not part of public API
        s.add(InvertedPredicateMetaOp.class);
        s.add(PostulateMetaOp.class);
        s.add(PredicateMetaOp.class);
        s.add(ReduceMetaOp.class);
        s.add(FreeGridOp.class);

        //Others that don't relaly make sense as a namespace method
        s.add(CopyOp.class);
        s.add(org.nd4j.linalg.api.ops.impl.transforms.pairwise.Set.class);
        s.add(PowPairwise.class);   //We have custom op Pow already used for this
        s.add(Create.class);    //Already have zeros, ones, etc for this
        s.add(HashCode.class);
        s.add(ScalarSetValue.class);
        s.add(PrintVariable.class);
        s.add(PrintAffinity.class);
        s.add(Assign.class);



    }

    @Test @Ignore
    public void generateOpClassList() throws Exception{
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends DifferentialFunction>> subTypes = reflections.getSubTypesOf(DifferentialFunction.class);

        List<Class<?>> l = new ArrayList<>();
        for(Class<?> c : subTypes){
            if(Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers()) )
                continue;
            l.add(c);
        }

        Collections.sort(l, new Comparator<Class<?>>() {
            @Override
            public int compare(Class<?> o1, Class<?> o2) {
                return o1.getName().compareTo(o2.getName());
            }
        });

        for(Class<?> c : l){
            System.out.println(c.getName() + ".class,");
        }
    }

}
