package org.nd4j.autodiff.samediff;

import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpExecOrder;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.layers.Linear;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.impl.UniformInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.util.*;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeFalse;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 4/11/17.
 */
public class SameDiffTests {
    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    public Map<String, INDArray> variablesForInput() {
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });

        INDArray labels = Nd4j.create(new double[]{1, 1, 0, 1}).reshape(4, 1);

        INDArray weights = Nd4j.zeros(3, 1);

        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put("x", inputs);
        inputMap.put("w", weights);
        inputMap.put("y", labels);
        return inputMap;
    }




    @Test
    public void testEvalVariable() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray twos = ones.add(ones);
        SDVariable inputOne = sameDiff.var("inputone", ones);
        SDVariable inputResult = inputOne.add("extravarname",inputOne);
        assertEquals(twos, inputResult.eval());
    }

    @Test
    public void testSigmoid() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = sameDiff.sigmoid(x);
        assertEquals(2, sameDiff.graph().numVertices());
        assertEquals(1, sameDiff.graph().getEdges().size());
        assertArrayEquals(arr.shape(), sigmoid.getShape());
        assertEquals(1, sameDiff.graph()
                .getVertexInDegree(sigmoid.getVertexId()));
        int[][] sorted = new int[][]{x.getVertexId(), sigmoid.getVertexId()};
        int[][] topoSortResult= sameDiff.graph().topologicalSort();
        assertArrayEquals(sorted, topoSortResult);
        assertEquals(1, sameDiff.graph().getOpOrder().getActions().size());
        OpState opState = sameDiff.graph().getOpOrder().getActions().get(0).getOpState();
        assertEquals("sigmoid", opState.getOpName());
        Op op = (Op) sameDiff.createOp(sameDiff.graph().getOpOrder().getActions().get(0));
        assertTrue(op instanceof Sigmoid);
        Nd4j.getExecutioner().exec(op);
        assertEquals(Transforms.sigmoid(Nd4j.linspace(1, 4, 4)), op.z());
    }

    @Test
    public void testSum() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.sum(x, 1);
        assertEquals(2, sameDiff.graph().numVertices());
        assertEquals(1, sameDiff.graph().getEdges().size());
        assertArrayEquals(arr.shape(), result.getShape());
        int[][] topoSortResult = sameDiff.graph().topologicalSort();
        assertArrayEquals(new int[][]{{1},{2}}, topoSortResult);
    }

    @Test
    public void testReshape() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.reshape(x, 2, 2);
        assertEquals(2, sameDiff.graph().numVertices());
        assertEquals(1, sameDiff.graph().getEdges().size());
        assertArrayEquals(new int[]{2, 2}, result.getShape());

    }

    @Test
    public void testTranspose() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.transpose(x);
        sameDiff.exec();
        assertEquals(2, sameDiff.graph().numVertices());
        assertEquals(1, sameDiff.graph().getEdges().size());
        assertArrayEquals(new int[]{4, 1}, result.getArr().shape());

    }


    @Test
    public void testDynamicOp() {
        SameDiff sameDiff = SameDiff.create();
        DynamicCustomOp dynamicCustomOp = DynamicCustomOp.
                sameDiffBuilder("testop",sameDiff)
                .addInputs(sameDiff.var("i1",new int[]{2,2}),
                        sameDiff.var("i2",new int[]{2,2}))
                .addOutputShape(new int[]{2,2})
                .addOutputShape(new int[]{2,3})
                .build();
        assertEquals(2,dynamicCustomOp.getOutputFunctions().length);


    }




    @Test
    public void testDistance() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.cosineSimilarity(x, y, 1);
        SDVariable addResult = result.add(result);

        assertEquals(4, sameDiff.graph().numVertices());
        assertArrayEquals(new int[]{1, 2}, result.getShape());
    }

    @Test
    public void testTensorGradMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.mmul(x, y);
        SDVariable otherResult = result.add(result);
        //2 inputs and 2 op results
        assertEquals(4, sameDiff.graph().numVertices());
        //2 edges for output
        assertEquals(2, sameDiff.graph().getEdges().size());
        assertArrayEquals(new int[]{2, 2}, result.getShape());
    }


    @Test
    public void testGetInputs() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.mmul(x, y);
        SDVariable otherResult = result.add(result);
        assertEquals(2, sameDiff.graph().getInputs().size());
    }

    @Test
    public void testEval() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = sameDiff.sigmoid(x);
        INDArray assertion = Transforms.sigmoid(arr);
        INDArray[] eval = sameDiff.eval(Collections.singletonMap("x", arr));
        assertEquals(assertion, eval[0]);

    }


    @Test
    public void testDefineFunctionVariableReferences() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable varOne = sameDiff.var("one",Nd4j.ones(2));
        String funcName = "func1";
        SameDiff func = sameDiff.defineFunction(funcName, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return variableInputs;
            }
        },new SDVariable[]{varOne});

        assertEquals(varOne.getVarName(),func.getVariable(varOne.getVarName()).getVarName());
        assertTrue(varOne.getArr() == func.getVariable(varOne.getVarName()).getArr());
    }




    @Test
    public void testCrossSameDiffVariableInitWithAlloc() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();


        SDVariable firstVar = first.var("one",new int[]{2,2});
        SDVariable secondVar = second.var(firstVar);
        assertTrue(firstVar.getArr() == secondVar.getArr());
        assertEquals(firstVar.getVarName(),secondVar.getVarName());

    }



    @Test
    public void testCrossSameDiffVariableInitWithPlaceHolder() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();


        SDVariable firstVar = first.var("one",new int[]{2,2});
        SDVariable secondVar = second.var(firstVar);
        assumeNotNull(firstVar.getArr());
        assertFalse("Vertex ids found to be the same reference",firstVar.getVertexId() == secondVar.getVertexId());
        assertTrue(firstVar.getArr() == secondVar.getArr());
        assertEquals(firstVar.getVarName(),secondVar.getVarName());

    }


    @Test
    public void testVariableArrayReference() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable arr = sameDiff.var("one",new int[]{2,2});
        assertArrayEquals(new int[]{2,2},arr.getShape());
        assumeNotNull(arr.getArr());
        assertArrayEquals(new int[]{2,2},arr.getArr().shape());

    }

    @Test
    public void testEvalAddSelf() {
        /**
         * Note this test fails yet due to needing
         * to validate simple cases like x * x
         * matching number of inputs.
         */
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = x.mul(x);
        INDArray assertion = arr.mul(arr);
        INDArray[] eval = sameDiff.eval(Collections.singletonMap("x", arr));
        assertEquals(assertion, eval[0]);

    }

    @Test
    public void testEvalAdd() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        INDArray yArr = arr.dup();
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", yArr);

        SDVariable sigmoid = x.mul(y);
        INDArray assertion = arr.mul(arr);
        Map<String, INDArray> vars = new HashMap<>();
        vars.put("x", arr);
        vars.put("y", yArr);
        INDArray[] eval = sameDiff.eval(vars);
        assertEquals(assertion, eval[0]);

    }


    @Test
    public void testTensorGradTensorMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.tensorMmul(x, y, new int[][]{{0}, {1}});
        assertEquals(3, sameDiff.graph().numVertices());
        assertEquals(1, sameDiff.graph().getEdges().size());
        assertArrayEquals(ArrayUtil.getTensorMmulShape(new int[]{2, 2, 2}, new int[]{2, 2, 2}, new int[][]{{0}, {1}}), result.getShape());
        assertEquals(48, sameDiff.numElements());
    }

    @Test
    public void testDup() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SameDiff tg2 = sameDiff.dup();

        assertEquals(sameDiff.graph(), tg2.graph());
    }


    @Test
    public void testLogGrad() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.var("x", Nd4j.linspace(1, 4, 4));
        SDVariable log = sameDiff.log(input);
        SDVariable sum = sameDiff.sum(log,Integer.MAX_VALUE);
        INDArray result = null;
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = sameDiff.execBackwards();
        System.out.println(execBackwards);
        //INDArray assertion = Nd4j.create(new double[]{1, 0.5, 0.33, 0.25});
        // assertTrue(assertion.equalsWithEps(result, 1e-2));
    }


    @Test
    public void testElementWiseDivAndRDiv() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray toDivBy = Nd4j.valueArrayOf(4, 0.25);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        xAndY.put("y", toDivBy);
        sameDiff.defineFunction("div", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[] {x.div(y)};
            }
        }, xAndY);

        sameDiff.defineFunction("rdiv", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[] {x.rdiv(y)};
            }
        }, xAndY);


        INDArray assertionForDiv = Nd4j.valueArrayOf(4, 4.0);
        INDArray assertionForRDiv = Nd4j.valueArrayOf(4, 0.25);
        assertEquals(assertionForDiv, sameDiff.getFunction("div").execAndEndResult());
        assertEquals(assertionForRDiv, sameDiff.getFunction("rdiv").execAndEndResult());

    }


    @Test
    public void testNegativeGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        sameDiff.defineFunction("neg", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                return new SDVariable[] {sameDiff.neg(x)};
            }
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, -1);
        assertEquals(assertionForDiv, sameDiff.getFunction("neg").execAndEndResult());

    }



    @Test
    public void testSumOp() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("sum", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable sum = sameDiff.sum(input, 1);
                return new SDVariable[] {sum};
            }
        }, inputs);

        INDArray assertion = sumInput.sum(1);
        INDArray executions = sameDiff.execAndEndResult("sum");
        assertEquals(assertion, executions);
    }


    @Test
    public void testVariableReferenceNoFunction() {
        /**
         * Creating a variable should not create a differential function.
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one",Nd4j.scalar(1.0));
        assumeNotNull(sameDiff.getVariableForVertexId(sdVariable.getVertexId()));
        assertFalse(sameDiff.getFunctionForVertexId(sdVariable.getVertexId()) != null);
    }


    @Test
    public void testVariableWithFunction() {
        /**
         * A variable's function should be null
         * when just a variable but
         * have a function result
         * when the variable itself is the result of a function.
         *
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one",Nd4j.scalar(1.0));
        SDVariable add = sdVariable.add(1.0);
        assertEquals(sameDiff.getVariableForVertexId(add.getVertexId()),add);
        assumeFalse(sameDiff.getFunctionForVertexId(sdVariable.getVertexId()) != null);
    }



    @Test
    public void testUpdateVariable() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable one = sameDiff.one("one",new int[]{1,1});
        sameDiff.updateVariableName(one.getVertexId(),"one-diff");
        assertEquals(one.getArr(),sameDiff.getVariable("one-diff").getArr());
    }


    @Test
    public void testMulGradient() {
        INDArray arr1 = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray arr2 = Nd4j.linspace(1,4,4).reshape(2,2);

        INDArray gradAssertion =  Nd4j.ones(arr1.shape());
        INDArray scalar = Nd4j.scalar(1.0);
        INDArray aGradAssertion = Nd4j.create(new double[][]{
                {1,4},
                {9,16}
        });

        INDArray cGradAssertion = Nd4j.create(new double[][]{
                {1,2},
                {3,4}
        });

        INDArray wGradAssertion = Nd4j.create(new double[][]{
                {2,8},
                {18,32}
        });

        INDArray dGradAssertion = Nd4j.ones(2,2);

        SameDiff sameDiff = SameDiff.create();

        SDVariable sdVariable = sameDiff.var("a",arr1);
        SDVariable sdVariable1 = sameDiff.var("w",arr2);
        SDVariable varMulPre = sdVariable.mul("c",sdVariable1);
        SDVariable varMul = varMulPre.mul("d",sdVariable1);
        SDVariable sum = sameDiff.sum("ret",varMul,Integer.MAX_VALUE);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> mapListPair = sameDiff.execBackwards();

        SDVariable finalResult = sameDiff.grad(sum.getVarName());

        SDVariable cGrad = sameDiff.grad(varMulPre.getVarName());

        SDVariable mulGradResult = sameDiff.grad(varMul.getVarName());
        SDVariable aGrad = sameDiff.grad(sdVariable.getVarName());
        SDVariable wGrad = sameDiff.grad(sdVariable1.getVarName());
        SDVariable dGrad = sameDiff.grad(varMul.getVarName());

        INDArray scalarGradTest = finalResult.getArr();
        assertEquals(scalar,scalarGradTest);


        INDArray gradTest = mulGradResult.getArr();
        assertEquals(gradAssertion,gradTest);

        INDArray aGradTest = aGrad.getArr();
        assertEquals(aGradAssertion,aGradTest);

        INDArray cGradTest = cGrad.getArr();
        assertEquals(cGradAssertion,cGradTest);

        INDArray wGradTest = wGrad.getArr();
        assertEquals(wGradAssertion,wGradTest);

        INDArray dGradTest = dGrad.getArr();
        assertEquals(dGradAssertion,dGradTest);


    }


    @Test
    public void testLinearModule() {
        int nIn = 5;
        Linear linear = Linear.execBuilder()
                .nIn(nIn)
                .nOut(4)
                .weightInitScheme(new UniformInitScheme('f',nIn))
                .biasWeightInitScheme(new ZeroInitScheme('f'))
                .build();
        linear.exec(Nd4j.linspace(1,20,20).reshape(4,5));
        assertEquals(1,linear.getOutputArguments().size());

    }



    @Test
    public void testInPlaceAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable toAdd = sameDiff.var("arr1",Nd4j.ones(2,2));
        SDVariable add = sameDiff.var("arr2",Nd4j.valueArrayOf(2,2,2.0));
        SDVariable result = toAdd.addi(add);
        sameDiff.execAndEndResult();
        INDArray arr = result.getArr();
        INDArray assertion = Nd4j.ones(2,2).addi(Nd4j.valueArrayOf(2,2,2.0));
        assertEquals(arr,assertion);
    }


    @Test
    public void multiInputOutputTest() {
        Graph<Integer,Integer> ints = new Graph<>();
        ints.addVertex(new Vertex<>(1,0,1));
        ints.addVertex(new Vertex<>(2,0,2));
        ints.addVertex(new Vertex<>(3,1,3));
        ints.addVertex(new Vertex<>(4,0,2));
        ints.addVertex(new Vertex<>(5,1,3));
        //multiple edge outputs
        ints.addEdge(new Edge<>(new int[]{1},new int[]{2},0,true));
        ints.addEdge(new Edge<>(new int[]{1},new int[]{3},0,true));

        ints.addEdge(new Edge<>(new int[]{2},new int[]{3},0,true));
        ints.addEdge(new Edge<>(new int[]{3},new int[]{2},0,true));

        assertEquals(2,ints.getEdgesOut(new int[]{1}).size());
        assertEquals(2,ints.getIncomingEdges().get(new int[]{3}).size());
    }



    @Test
    public void testDefineFunctionArrayExistence() {
        SameDiff sameDiff = SameDiff.create();
        String testFunctionName = "testfunction";
        SDVariable[] inputVars = new SDVariable[] {
                sameDiff.var("one",new int[]{1,1}),
                sameDiff.var("two",new int[]{1,1}),

        };

        SameDiff functionDef = sameDiff.defineFunction(testFunctionName, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return new SDVariable[] {variableInputs[0].add(variableInputs[1])};
            }
        },inputVars);


        //1 input plus 2 outputs
        assertEquals(4,functionDef.variables().size());



    }


    @Test
    public void testWhileLoop() {
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0],variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0],ret};
            }
        },new SDVariable[] {
                sameDiff.one("one",new int[]{1,1}),
                sameDiff.var("two",new int[]{1,1}),

        });

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();
        assertFalse(exec.getRight().isEmpty());
        While function = (While) exec.getRight().get(exec.getRight().size() - 1);
        assumeNotNull(function.getOutputVars());
        assertEquals(1,function.getNumLooped());
        sameDiff.toString();
    }



    @Test
    public void testIfStatementTrueBodyBackwards() {
        SameDiff sameDiff = SameDiff.create();
        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0],Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum,1.0);
                return new SDVariable[] {result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[] {add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody =  new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[] {sub};
            }
        };

        //true body trigger
        SDVariable[] firstInputs = new SDVariable[] {
                sameDiff.var("one",new int[]{1,1})

        };



        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody,firstInputs);
        sameDiff.execBackwards();
        SameDiff grad = sameDiff.getFunction("grad");
        If ifBlock = (If) grad.getFunctionForVertexId(new int[]{2});
        SameDiff assertComparision = SameDiff.create();
        SDVariable initialInput = assertComparision.zero("zero",new int[]{1,1});
        initialInput.addi(1.0);
        assumeNotNull(ifBlock.getTrueBodyExecuted());
        assertTrue(ifBlock.getTrueBodyExecuted());
        assertEquals(Nd4j.scalar(1.00),initialInput.getArr());
        assertEquals(Nd4j.scalar(1.0),ifBlock.getLoopBodyExecution().getArrForVertexId(2));

    }


    @Test
    public void testConv2d() {

    }

    @Test
    public void testWhileBackwards() {
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0],variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0],ret};
            }
        },new SDVariable[] {
                sameDiff.one("one",new int[]{1,1}),
                sameDiff.var("two",new int[]{1,1}),

        });

        sameDiff.execBackwards();
        SameDiff exec = sameDiff.getFunction("grad");
        System.out.println(exec);
    }


    @Test
    public void testIfStatementTrueBody() {
        SameDiff sameDiff = SameDiff.create();

        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0],Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum,1.0);
                return new SDVariable[] {result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[] {add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody =  new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[] {sub};
            }
        };

        //true body trigger
        SDVariable[] firstInputs = new SDVariable[] {
                sameDiff.var("one",new int[]{1,1})

        };



        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody,firstInputs);
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();

    }


    @Test
    public void testIfStatementFalseBody() {
        SameDiff sameDiff = SameDiff.create();

        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0],Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum,1.0);
                return new SDVariable[] {result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[] {add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody =  new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[] {sub};
            }
        };


        //false body trigger
        SDVariable[] secondInputs = new SDVariable[] {
                sameDiff.setupFunction(sameDiff.var("two",new int[]{1,1}))

        };

        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody,secondInputs);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();




    }


    @Test
    public void testAutoBroadcastAddMatrixector() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row = Nd4j.ones(2);
        INDArray assertion = arr.add(1.0);
        SDVariable left = sameDiff.var("arr",arr);
        SDVariable right = sameDiff.var("row",row);
        SDVariable test = left.add(right);
        sameDiff.exec();
        assertEquals(assertion,test.getArr());
    }

    @Test
    public void testRunLogisticRegression() {
        Map<String,INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable activation = sameDiff.sigmoid("activation",sameDiff.mmul("mmul",x,w));
                SDVariable oneMinusY = y.rsub("oneminusy",1.0);
                SDVariable oneMinusPredictions = activation.rsub("oneminusactivations",1.0);
                SDVariable outputTimesY = y.mul("output * y",activation);
                SDVariable yHat = oneMinusPredictions.mul("yhat",oneMinusY);
                SDVariable probs = outputTimesY.add("probs",yHat);
                SDVariable logProbs = sameDiff.log("logprob",probs);
                SDVariable ret = sameDiff.sum("totalsum",logProbs,Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum",ret);
                return new SDVariable[] {ret2};
            }
        },vars);

        SameDiff activation = outside.getFunction("activate");
        int epochsToRun = 5;
        double lr = 0.1;
        for(int i = 0; i < epochsToRun; i++) {
            activation.execBackwards();
            INDArray wGrad = activation.grad("w").getArr().reshape(vars.get("w").shape());
            vars.get("w").subi(wGrad.mul(lr));
            System.out.println("Score: " + activation.getVariable("negtotalsum").getArr());
        }

    }


    @Test
    public void testSoftmaxRegression() {
        Map<String,INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable activation = sameDiff.softmax("activation",sameDiff.mmul("mmul",x,w));
                SDVariable ret = sameDiff.sum("totalsum",activation,Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum",ret);
                return new SDVariable[] {ret2};
            }
        },vars);


        /**
         * Backwards should be:
         * neg score
         * sum sum of log
         * log (log probs)
         * add
         * mul
         * mul
         * rsub (predictions)
         * sigmoid
         * rsub
         * matrix multiply
         *
         */


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> opsBackward = outside.getFunction("activate").execBackwards();
        SameDiff gradSameDiff = outside.getFunction("activate").getFunction("grad");

        SDVariable gradWrtX = outside.getFunction("activate").grad("x");
        SDVariable gradWrtW = outside.getFunction("activate").grad("w");
        assumeNotNull(gradWrtX);
        assumeNotNull(gradWrtW);

        INDArray wGradAssertion = Nd4j.create(new double[]{0,0,0}).reshape(3,1);
        assertEquals(wGradAssertion,outside.getFunction("activate").grad("w").getArr());
        //note here that the gradients here end up being some weird really low eps where it
        //isn't exactly zero
        //        assertEquals(inputAssertion,outside.getFunction("activate").grad("x").getArr());



        System.out.println(gradWrtX);
        System.out.println(gradWrtW);


    }

    @Test
    public void testLogisticRegression() {
        Map<String,INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable activation = sameDiff.sigmoid("activation",sameDiff.mmul("mmul",x,w));
                SDVariable oneMinusY = y.rsub("oneminusy",1.0);
                SDVariable oneMinusPredictions = activation.rsub("oneminusactivations",1.0);
                SDVariable outputTimesY = y.mul("output * y",activation);
                SDVariable yHat = oneMinusPredictions.mul("yhat",oneMinusY);
                SDVariable probs = outputTimesY.add("probs",yHat);
                SDVariable logProbs = sameDiff.log("logprob",probs);
                SDVariable ret = sameDiff.sum("totalsum",logProbs,Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum",ret);
                return new SDVariable[] {ret2};
            }
        },vars);


        /**
         * Backwards should be:
         * neg score
         * sum sum of log
         * log (log probs)
         * add
         * mul
         * mul
         * rsub (predictions)
         * sigmoid
         * rsub
         * matrix multiply
         *
         */


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> opsBackward = outside.getFunction("activate").execBackwards();
        SameDiff gradSameDiff = outside.getFunction("activate").getFunction("grad");

        SDVariable gradWrtX = outside.getFunction("activate").grad("x");
        SDVariable gradWrtW = outside.getFunction("activate").grad("w");
        assumeNotNull(gradWrtX);
        assumeNotNull(gradWrtW);

        INDArray wGradAssertion = Nd4j.create(new double[]{-0.81,1.255,-1.80499983}).reshape(3,1);
        INDArray inputAssertion = Nd4j.valueArrayOf(vars.get("x").shape(),1e-1);
        INDArray yGradAssertion = Nd4j.zeros(vars.get("y").shape());
        INDArray mmulGrad = Nd4j.create(new double[]{-0.5,-0.5,0.5,-0.5}).reshape(4,1);
        INDArray predsGradAssertion = Nd4j.create(new double[]{-2,-2,2,-2}).reshape(4,1);
        INDArray oneMinusPredsGradAssertion = Nd4j.create(new double[]{0,0,-2,0}).reshape(4,1);
        INDArray oneMinusLabelsAssertion = Nd4j.valueArrayOf(4,-1).reshape(4,1);
        INDArray outputTimesYGradAssertion = Nd4j.valueArrayOf(4,-2).reshape(4,1);
        INDArray yHatAssertion = outputTimesYGradAssertion.dup();
        INDArray labelProbsGradAssertion = yHatAssertion.dup();
        INDArray logProbsGradAssertion = Nd4j.valueArrayOf(4,-1).reshape(4,1);

        assertEquals(logProbsGradAssertion,outside.getFunction("activate").grad("logprob").getArr());
        assertEquals(labelProbsGradAssertion,outside.getFunction("activate").grad("probs").getArr());
        assertEquals(yHatAssertion,outside.getFunction("activate").grad("yhat").getArr());
        assertEquals(outputTimesYGradAssertion,outside.getFunction("activate").grad("output * y").getArr());
        assertEquals(oneMinusLabelsAssertion,outside.getFunction("activate").grad("oneminusy").getArr());
        assertEquals(oneMinusPredsGradAssertion,outside.getFunction("activate").grad("oneminusactivations").getArr());
        assertEquals(predsGradAssertion,outside.getFunction("activate").grad("activation").getArr());
        assertEquals(mmulGrad,outside.getFunction("activate").grad("mmul").getArr());
        assertEquals(yGradAssertion,outside.getFunction("activate").grad("y").getArr());
        assertEquals(wGradAssertion,outside.getFunction("activate").grad("w").getArr());
        //note here that the gradients here end up being some weird really low eps where it
        //isn't exactly zero
        //        assertEquals(inputAssertion,outside.getFunction("activate").grad("x").getArr());



        System.out.println(gradWrtX);
        System.out.println(gradWrtW);


    }



    @Test
    public void testNestedExecution() {
        final SameDiff outer = SameDiff.create();
        Map<String, INDArray> input = new HashMap<>();
        input.put("x", Nd4j.ones(2));
        outer.defineFunction("firstadd", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable ret = input.add(input);
                return new SDVariable[] {ret};
            }
        }, input);

        outer.defineFunction("secondadd", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable result = outer.invokeFunctionOn("firstadd", sameDiff);
                return new SDVariable[] {result.add(1.0)};
            }
        });

        SameDiff secondAdd = outer.getFunction("secondadd");
        INDArray[] outputs = secondAdd.eval(input);
        INDArray outputsAssertion = Nd4j.valueArrayOf(2, 2.0);
        assertEquals(outputsAssertion, outputs[0]);
    }


    @Test
    public void testResultPropagation() {
        SameDiff sameDiff = SameDiff.create();
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });


        INDArray weights = Nd4j.randn(3, 1);

        SDVariable x = sameDiff.var("x", inputs);
        SDVariable w = sameDiff.var("w", weights);
        SDVariable preOutput = sameDiff.mmul(x, w);

        SDVariable outputs = sameDiff.sigmoid(preOutput);
        List<DifferentialFunction> ops = sameDiff.exec().getRight();
        Op firstOp = (Op) ops.get(0);
        Op secondOp = (Op) ops.get(1);
        assertTrue(firstOp.z() == secondOp.x());

    }

    @Test
    public void testSimpleDefineFunction() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        inputs.remove("y");
        String logisticForward = "logisticPredictions";
        sameDiffOuter.defineFunction(logisticForward, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {

                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);
                return new SDVariable[] {sigmoid};
            }

        }, inputs);

        assertEquals(1, sameDiffOuter.definedFunctionNames().size());

        //note here that we don't add the duplicate ops with define function anymore
    }


    @Test
    public void testSoftmax() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("softmax", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x").dup());
                SDVariable softmax = sameDiff.softmax(input);
                //original shape ends up being 2,2
                return new SDVariable[] {softmax};
            }
        }, inputs);

        INDArray executions = sameDiff.execAndEndResult("softmax");
        INDArray assertions = Transforms.softmax(sumInput.dup());
        assertArrayEquals(sumInput.shape(), executions.shape());
        System.out.println(executions);
        assertEquals(assertions, executions);


        SoftMaxDerivative softMaxDerivative = new SoftMaxDerivative(sumInput);
        Nd4j.getExecutioner().exec(softMaxDerivative);
        System.out.println(softMaxDerivative.z());
    }




    @Test
    public void testSigmoidBackwards() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        SDVariable input = sameDiff.var("x",inputs.get("x"));
        SDVariable sigmoid = sameDiff.sigmoid(input);
        SDVariable sum = sameDiff.sum(sigmoid,Integer.MAX_VALUE);
        List<DifferentialFunction> backwardsOps = sameDiff.execBackwards().getRight();
        Op finalOp = (Op)  backwardsOps.get(backwardsOps.size() - 1);
        assertTrue(Nd4j.create(new double[][]{
                {0.1966 , 0.1050},
                {0.0452 , 0.0177}
        }).equalsWithEps(
                finalOp.z(),1e-2));
        System.out.println(backwardsOps);
    }


    @Test
    public void testSumGradient() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("initial",Nd4j.linspace(1,4,4).reshape(2,2));
        SDVariable sum = sameDiff.sum(twoByTwo,Integer.MAX_VALUE);
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = sameDiff.execBackwards();
        SameDiff grad = sameDiff.getFunction("grad");
        SDVariable gradArr = sameDiff.grad(twoByTwo.getVarName());
        assertEquals(Nd4j.ones(2,2),gradArr.getArr());
    }






    @Test
    public void testMmulGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1,4,4).reshape(2,2);
        Map<String,INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        inputs.put("y",sumInput.dup());

        sameDiff.defineFunction("mmulGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable input2 = sameDiff.var("y",inputs.get("y"));
                SDVariable exp = sameDiff.mmul(input,input2);
                SDVariable sum = sameDiff.sum(exp,Integer.MAX_VALUE);
                return new SDVariable[] {sum};
            }
        },inputs);

        List<DifferentialFunction> ops = sameDiff.getFunction("mmulGradient").execBackwards().getRight();

        assumeNotNull(sameDiff.getFunction("mmulGradient").getFunction("grad"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("x"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("y"));

        SDVariable gradWrtX = sameDiff.getFunction("mmulGradient").grad("x");
        SDVariable gradWrtY = sameDiff.getFunction("mmulGradient").grad("y");
        assumeNotNull(gradWrtX.getArr());
        assumeNotNull(gradWrtY.getArr());


        INDArray xGradAssertion = Nd4j.create(new double[][]{
                {3,7},
                {3,7}
        });

        INDArray yGradAssertion = Nd4j.create(new double[][]{
                {4,4},
                {6,6}
        });

        assertEquals(xGradAssertion,gradWrtX.getArr());
        assertEquals(yGradAssertion,gradWrtY.getArr());

    }

    @Test
    public void testExpGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1,4,4).reshape(2,2);
        Map<String,INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        sameDiff.defineFunction("expGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable exp = sameDiff.exp(input);
                SDVariable sum = sameDiff.sum(exp,Integer.MAX_VALUE);
                return new SDVariable[] {sum};
            }
        },inputs);


        List<DifferentialFunction> ops = sameDiff.getFunction("expGradient").execBackwards().getRight();

        Op op = (Op) ops.get(ops.size() - 1);
        INDArray executions = op.z();
        INDArray assertion = Nd4j.create(new double[][]{
                {2.7183  , 7.3891},
                {20.0855  ,54.5981}
        });
        assertArrayEquals(sumInput.shape(),executions.shape());
        assertEquals(assertion,executions);
        System.out.println(executions);
        //assertEquals(Nd4j.ones(2,2),executions);
    }


    @Test
    public void testDepth() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable x = sameDiff.one("one",new int[]{2,2});
        assertEquals(0,x.depth());
        SDVariable sigmoid = sameDiff.sigmoid("sigmoid",x);
        assertEquals(1,sigmoid.depth());
    }


    @Test
    public void testTanhGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1,4,4).reshape(2,2);
        Map<String,INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        sameDiff.defineFunction("tanhGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable tanh = sameDiff.tanh(input);
                SDVariable sum = sameDiff.sum(tanh,Integer.MAX_VALUE);
                return new SDVariable[] {tanh};
            }
        },inputs);

        INDArray executions = sameDiff.getFunction("tanhGradient").execBackwardAndEndResult();
        //[0.41997434161402614,0.07065082485316443,0.009866037165440211,0.0013409506830258655]
        INDArray assertion = Nd4j.create(new double[][]{
                {0.41997434161402614 , 0.07065082485316443},
                {0.009866037165440211 , 0.0013409506830258655}
        });

        assertTrue(assertion.equalsWithEps(
                executions,1e-3));

        assertArrayEquals(sumInput.shape(),executions.shape());
        assertEquals(assertion,executions);
        System.out.println(executions);
        //assertEquals(Nd4j.ones(2,2),executions);
    }



    @Test
    public void testRsubScalar() {
        SameDiff sameDiff = SameDiff.create();
        Map<String,INDArray> params = new HashMap<>();
        INDArray var = Nd4j.valueArrayOf(4,2);
        params.put("x",var);
        sameDiff.defineFunction("rsubop", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable ret = input.rsub(1.0);
                return new SDVariable[] {ret};
            }
        },params);

        SameDiff logisticGraph = sameDiff.getFunction("rsubop");
        INDArray[] outputs = logisticGraph.eval(params);
        assertEquals(Nd4j.ones(4).muli(-1),outputs[0]);
        System.out.println(Arrays.toString(outputs));



    }

    @Test
    public void testFunctionScalarResultPropagation() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input,w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);
                return new SDVariable[] {sigmoid};
            }
        },inputs);

        sameDiffOuter.defineFunction("oneminuspredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable oneMinusPredictions = y.rsub(1.0);
                return new SDVariable[] {oneMinusPredictions};
            }
        },inputs);


        SameDiff logisticGraph = sameDiffOuter.getFunction("oneminuspredictions");
        INDArray[] outputs = logisticGraph.eval(inputs);
        INDArray assertion = Nd4j.create(new double[]{0,0,1,0});
        assertEquals(assertion,outputs[outputs.length - 1]);

    }

    @Test
    public void testInplaceSubi() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> params = new HashMap<>();
        params.put("x",Nd4j.ones(4));
        sameDiffOuter.defineFunction("inplacesubi", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable inplace = sameDiff.var("x",inputs.get("x"));
                return new SDVariable[] {inplace.subi(1.0)};
            }
        },params);

        sameDiffOuter.getFunction("inplacesubi").eval(params);
        assertEquals(Nd4j.zeros(4),params.get("x"));
    }


    @Test
    public void testMmul() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> inputs = variablesForInput();
        SDVariable x = sameDiffOuter.var("x",inputs.get("x"));
        SDVariable w = sameDiffOuter.var("w",inputs.get("w"));
        SDVariable output = sameDiffOuter.mmul(x,w);
        assertEquals(1,sameDiffOuter.graph().getOpOrder().getActions().size());
    }




    @Test
    public void testGraphBuildingWithScalars() {
        final SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions",new LogisticPredictions(),inputs);

        SameDiff logisticPrediction = sameDiffOuter.getFunction("logisticPredictions");
        List<String> logisticOpNameAssertions = Arrays.asList("mmul","sigmoid");
        //do standalone test before new op definition to verify graph references
        //aren't changed with new instances
        OpExecOrder logisticPredictionOrder = logisticPrediction.graph().getOpOrder();
        for(int i = 0; i < 2; i++) {
            assertEquals(logisticOpNameAssertions.get(i),logisticPredictionOrder.getActions().get(i).getOpState().getOpName());
        }


        sameDiffOuter.defineFunction("loss", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable outputs = sameDiffOuter.invokeFunctionOn("logisticPredictions",sameDiff);
                SDVariable outputTimesY = outputs.rsub(1.0);
                return new SDVariable[] {outputTimesY};
            }
        },inputs);


        logisticPredictionOrder = logisticPrediction.graph().getOpOrder();
        for(int i = 0; i < 2; i++) {
            assertEquals(logisticOpNameAssertions.get(i),logisticPredictionOrder.getActions().get(i).getOpState().getOpName());
        }

        SameDiff logisticGraph = sameDiffOuter.getFunction("loss");
        List<String> opNameAssertions = Arrays.asList("mmul","sigmoid","rsub_scalar");
        OpExecOrder opExecOrder = logisticGraph.graph().getOpOrder();
        System.out.println(opExecOrder);
        assertEquals(3,opExecOrder.getActions().size());
        for(int i = 0; i < 3; i++) {
            assertEquals(opNameAssertions.get(i),opExecOrder.getActions().get(i).getOpState().getOpName());
        }

    }


    @Test
    public void testGraphBuilding() {
        final SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);

                return new SDVariable[]{sigmoid};
            }
        }, inputs);

        sameDiffOuter.defineFunction("loss", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable outputs = sameDiffOuter.invokeFunctionOn("logisticPredictions", sameDiff);
                SDVariable y = sameDiff.getVariable("y");
                SDVariable outputTimesY = outputs.mul(y);
                return new SDVariable[]{outputTimesY};

            }
        }, inputs);
        {


            SameDiff logisticPrediction = sameDiffOuter.getFunction("logisticPredictions");
            List<String> logisticOpNameAssertions = Arrays.asList("mmul", "sigmoid");
            OpExecOrder logisticPredictionOrder = logisticPrediction.graph().getOpOrder();
            for (int i = 0; i < 2; i++) {
                assertEquals(logisticOpNameAssertions.get(i), logisticPredictionOrder.getActions().get(i).getOpState().getOpName());
            }

            SameDiff logisticGraph = sameDiffOuter.getFunction("loss");
            List<String> opNameAssertions = Arrays.asList("mmul", "sigmoid", "mul");
            OpExecOrder opExecOrder = logisticGraph.graph().getOpOrder();
            assertEquals(3, opExecOrder.getActions().size());
            for (int i = 0; i < 3; i++) {
                assertEquals(opNameAssertions.get(i), opExecOrder.getActions().get(i).getOpState().getOpName());
            }

        }
    }


    @Test
    public void testScalarAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("first",Nd4j.linspace(1,4,4).reshape(2,2));
        SDVariable add = twoByTwo.add(1.0);
        INDArray test = sameDiff.execAndEndResult();
        INDArray assertion = Nd4j.linspace(1,4,4).reshape('f',2,2).add(1.0);
        assertEquals(assertion,test);
    }


    @Test
    public void testSums() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones",ones);
        SDVariable result = sdVariable.addi(1.0);
        SDVariable total = sameDiff.sum(result,Integer.MAX_VALUE);
        List<DifferentialFunction> ops = sameDiff.exec().getRight();
        INDArray output = null;
        for(int i = 0; i < 5; i++) {
            output = sameDiff.execAndEndResult(ops);
            System.out.println("Ones " + ones);
            System.out.println(output);
        }

        assertEquals(Nd4j.valueArrayOf(4,7),ones);
        assertEquals(28,output.getDouble(0),1e-1);
    }

}

