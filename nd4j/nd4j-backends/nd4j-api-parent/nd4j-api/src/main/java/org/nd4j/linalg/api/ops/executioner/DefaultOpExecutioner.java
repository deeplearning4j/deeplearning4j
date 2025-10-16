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

package org.nd4j.linalg.api.ops.executioner;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.any.Assign;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Optional;
import org.nd4j.linalg.profiler.*;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.array.event.NDArrayMetaData;
import org.nd4j.linalg.profiler.data.array.eventlog.DefaultNd4jEventLog;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.nativeblas.OpaqueShapeList;
import org.nd4j.nativeblas.OpaqueVariable;
import org.nd4j.nativeblas.OpaqueVariablesSet;

import java.util.*;

@Slf4j
public abstract class DefaultOpExecutioner implements OpExecutioner {

    private static final String SCOPE_PANIC_MSG = "For more details, see the ND4J User Guide: https://deeplearning4j.konduit.ai/nd4j/reference#workspaces-scope-panic";
    public static Nd4jEventLog eventLog = new DefaultNd4jEventLog();

    protected ProfilingMode profilingMode = ProfilingMode.SCOPE_PANIC;
    protected ProfilerConfig profilerConfig  = ProfilerConfig.builder().build();
    protected AtomicBoolean verbose = new AtomicBoolean(false);
    protected AtomicBoolean debug = new AtomicBoolean(false);

    protected ThreadLocal<OpContext> nextOpContext = new ThreadLocal<>();


    public DefaultOpExecutioner() {}


    /**
     * Inject an op context created using
     * {@link #buildContext()}
     * and return a reference to the context.
     * @return
     */
    @Override
    public OpContext injectNewContext() {
        clearOpContext();
        OpContext opContext = buildContext();
        nextOpContext.set(opContext);
        return opContext;
    }

    /**
     * Clears the context injected
     * with {@link #injectNewContext()} ()}
     */
    @Override
    public void clearOpContext() {
        nextOpContext.remove();
    }

    /**
     * Setting an {@link OpContext} will cause
     * {@link #buildContext()} to consume the specified op context
     * in place of creating a  new one.
     * @param context
     */
    @Override
    public void setNextOpContext(OpContext context) {
        nextOpContext.set(context);
    }

    /**
     * Execute a redirected {@link org.nd4j.linalg.api.ops.impl.transforms.custom.Assign} op
     * from the old {@link TransformOp} based {@link Assign}
     * based Assign op
     * @param op the input op
     * @param oc the op context
     * @param executioner the op executioner
     */
    public static void execAssign(TransformOp op, OpContext oc, OpExecutioner executioner) {
        org.nd4j.linalg.api.ops.impl.transforms.custom.Assign op2 = new org.nd4j.linalg.api.ops.impl.transforms.custom.Assign();
        DifferentialFunction differentialFunction = (DifferentialFunction) op;
        op2.setSameDiff(differentialFunction.getSameDiff());
        if(oc == null) {
            if(Nd4j.getEnvironment().isDebugAndVerbose() && op.x().isView()) {
                log.warn("Assign op running on a view. This may cause issues with the underlying buffer being modified and the view not seeing these changes");
            }
            op2.addBArgument(op.x().isView());
            op2.addInputArgument(op.x());
            if(op.y() != null)
                op2.addInputArgument(op.y());
            else op2.addInputArgument(op.x());
            op2.addOutputArgument(op.z());
            INDArray[] result = executioner.exec(op2);
        } else {
            executioner.exec(op2, oc);

        }

    }


    /**
     *
     * @param op
     * @param shapeOverride
     * @param context
     */
    public static void initOpContext(CustomOp op, boolean shapeOverride, OpContext context) {
        // optionally skip shape validation on op execution
        if (shapeOverride)
            context.shapeFunctionOverride(true);

        context.markInplace(op.isInplaceCall());

        // transferring rng state
        context.setRngStates(Nd4j.getRandom().rootState(), Nd4j.getRandom().nodeState());

        //transferring input/output arrays
        context.setInputArrays(op.inputArguments());
        if(!op.isInplaceCall())
            context.setOutputArrays(op.outputArguments());


        // transferring static args
        context.setBArguments(op.bArgs());
        context.setIArguments(op.iArgs());
        context.setTArguments(op.tArgs());
        context.setDArguments(op.dArgs());
    }


    protected void checkForCompression(Op op) {
        if (op.x() != null && op.x().isCompressed())
            Nd4j.getCompressor().decompressi(op.x());

        if (op.y() != null && op.y().isCompressed())
            Nd4j.getCompressor().decompressi(op.y());

        if (op.z() != null && op.z().isCompressed())
            Nd4j.getCompressor().decompressi(op.z());
    }

    @Override
    public String getLastOp() {
        return "UNKNOWN";
    }



    @Override
    public abstract INDArray exec(Op op);

    @Override
    public abstract INDArray exec(Op op, OpContext opContext);

    @Override
    public Op execAndReturn(Op op) {
        if (op instanceof TransformOp) {
            return execAndReturn((TransformOp) op);
        }
        if (op instanceof ScalarOp) {
            return execAndReturn((ScalarOp) op);
        }
        if (op instanceof ReduceOp) {
            exec((ReduceOp) op);
            return op;
        }
        if (op instanceof IndexAccumulation) {
            exec((IndexAccumulation) op);
            return op;
        }

        throw new IllegalArgumentException("Illegal opType of op: " + op.getClass());
    }

    @Override
    public TransformOp execAndReturn(TransformOp op) {
        exec(op);
        return op;
    }


    @Override
    public ReduceOp execAndReturn(ReduceOp op) {
        exec(op);
        return op;
    }

    @Override
    public Variance execAndReturn(Variance op) {
        exec(op);
        return op;
    }

    @Override
    public ScalarOp execAndReturn(ScalarOp op) {
        exec(op);
        return op;
    }

    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        exec(op);
        return op;
    }

    @Override
    public BroadcastOp execAndReturn(BroadcastOp op) {
        exec(op);
        return op;
    }

    public abstract INDArray createFromDescriptor(DataBuffer shapeInformation);

    @Override
    public INDArray[] exec(CustomOp op) {
        return execAndReturn(op).outputArguments().toArray(new INDArray[0]);
    }

    @Override
    public abstract INDArray exec(ReduceOp op);

    @Override
    public abstract INDArray exec(Variance accumulation);

    @Override
    public abstract INDArray exec(IndexAccumulation op);

    @Override
    public abstract INDArray exec(BroadcastOp broadcast);

    @Override
    public void exec(MetaOp op) {
        throw new UnsupportedOperationException("MetaOp execution isn't supported for this OpExecutioner yet");
    }

    @Override
    public void exec(GridOp op) {
        throw new UnsupportedOperationException("GridOp execution isn't supported for this OpExecutioner yet");
    }


    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return exec(op, Nd4j.getRandom());
    }

    /**
     * This method executes specific RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    @Override
    public abstract INDArray exec(RandomOp op, Random rng);


    @Deprecated
    @Override
    public void setProfilingMode(ProfilingMode mode) {

        profilingMode = mode;
        switch (profilingMode) {
            case ALL:
                this.profilerConfig = ProfilerConfig.builder().checkWorkspaces(true).checkElapsedTime(true).stackTrace(true).build();
                break;
            case METHODS:
                this.profilerConfig = ProfilerConfig.builder().stackTrace(true).build();
                break;
            case OPERATIONS:
                this.profilerConfig = ProfilerConfig.builder().stackTrace(true).checkElapsedTime(true).build();
                break;
            case SCOPE_PANIC:
                this.profilerConfig = ProfilerConfig.builder().checkWorkspaces(true).build();
                break;
            case ANY_PANIC:
                this.profilerConfig = ProfilerConfig.builder().checkForINF(true).checkForNAN(true).build();
                break;
            case INF_PANIC:
                this.profilerConfig = ProfilerConfig.builder().checkForINF(true).build();
                break;
            case NAN_PANIC:
                this.profilerConfig = ProfilerConfig.builder().checkForNAN(true).build();
                break;
            default:
                this.profilerConfig = ProfilerConfig.builder().build();
                break;
        }

    }

    @Override
    public void setProfilingConfig(ProfilerConfig config) {
        this.profilerConfig = config;
    }

    @Deprecated
    @Override
    public ProfilingMode getProfilingMode() {
        return profilingMode;
    }

    protected void checkWorkspace(String opName, INDArray array) {
        if (array.isAttached() && !array.isView()) {
            val ws = array.data().getParentWorkspace();

            if (ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {

                if (!ws.isScopeActive()) {
                    throw new ND4JIllegalStateException("Op [" + opName + "] X argument uses leaked workspace pointer from workspace ["
                            + ws.getId() + "]: Workspace the array was defined in is no longer open.\nAll open workspaces: " + allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG
                            + " with workspace enum: " + ws.getAssociatedEnumType());
                }

                if (ws.getGenerationId() != array.data().getGenerationId())
                    throw new ND4JIllegalStateException("Op [" + opName + "] X argument uses outdated workspace pointer from workspace ["
                            + ws.getId() + "]: Workspace array was defined in has been closed and reopened at least once since array creation. Array WS iteration: " +
                            array.data().getGenerationId() + ". Workspace current iteration: " +
                            ws.getGenerationId() + "\nAll open workspaces: " + allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
            }
        }
    }

    protected void checkForWorkspaces(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();
        for (int i = 0; i < inArgs.size(); i++) {
            checkWorkspace(op.opName(), inArgs.get(i));
        }
        for (int i = 0; i < outArgs.size(); i++) {
            checkWorkspace(op.opName(), outArgs.get(i));
        }
    }

    protected void checkForWorkspaces(Op op, OpContext oc) {
        val x = oc != null ? oc.getInputArray(0) : op.x();
        if (x != null)
            checkWorkspace(op.opName(), x);

        val y = oc != null && oc.getInputArrays().size() > 1 ? oc.getInputArray(1) : op.y();
        if (y != null)
            checkWorkspace(op.opName(), y);

        val z = oc != null ? oc.getOutputArray(0) : op.z();
        if (z != null)
            checkWorkspace(op.opName(), z);
    }

    public static List<String> allOpenWorkspaces() {
        List<MemoryWorkspace> l = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        List<String> workspaces = new ArrayList<>(l.size());
        for(MemoryWorkspace ws : l) {
            if(ws.isScopeActive()) {
                workspaces.add(ws.getId());
            }
        }
        return workspaces;
    }

    @Deprecated
    public long profilingHookIn(Op op, DataBuffer... tadBuffers) {
        if(profilerConfig != null) {
            if(profilerConfig.isCheckForINF()) {
                OpExecutionerUtil.checkForNaN(op.z());
            } else if(profilerConfig.isCheckForNAN()) {
                OpExecutionerUtil.checkForNaN(op.z());
            }
        }
        return 0L;
    }

    @Deprecated
    public long profilingHookIn(CustomOp op, OpContext oc) {
        if(profilerConfig != null) {
            if(profilerConfig.isCheckForINF()) {
                OpExecutionerUtil.checkForNaN(op,oc);
            } else if(profilerConfig.isCheckForNAN()) {
                OpExecutionerUtil.checkForNaN(op,oc);
            }
        }
        return 0L;
    }

    @Deprecated
    public void profilingHookOut(Op op, OpContext oc, long timeStart) {
        if(Nd4j.getEnvironment().isLogNDArrayEvents()) {
            INDArray x = op.x() != null ? op.x() : oc.getInputArray(0);
            INDArray y = op.y() != null ? op.y() : oc.getInputArrays().size() >  1 ? oc.getInputArray(1) : null;
            INDArray z = op.z() != null ? op.z() : oc.getOutputArray(0);

            List<INDArray> inArgs = new ArrayList<>();
            if(x != null) {
                inArgs.add(x);
            }

            if(y != null) {
                inArgs.add(y);
            }

            z.addEvent(NDArrayEvent.builder()
                    .dataAtEvent(NDArrayMetaData.from(z))
                    .parentDataAtEvent(NDArrayMetaData.fromArr(inArgs))
                    .ndArrayEventType(NDArrayEventType.BEFORE_OP_OUTPUT)
                    .stackTrace(Thread.currentThread().getStackTrace())
                    .build());

            if(x != null) {
                INDArray arr = x;
                NDArrayEvent event = NDArrayEvent.builder()
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .dataAtEvent(NDArrayMetaData.from(arr))
                        .parentDataAtEvent(NDArrayMetaData.fromArr(arr))
                        .ndArrayEventType(NDArrayEventType.OP_INPUT)
                        .build();
                arr.addEvent(event);


            }

            if(y != null) {
                INDArray arr =  y;
                NDArrayEvent event = NDArrayEvent.builder()
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .parentDataAtEvent(NDArrayMetaData.fromArr(arr))
                        .dataAtEvent(NDArrayMetaData.from(arr))
                        .ndArrayEventType(NDArrayEventType.OP_INPUT)
                        .build();
                arr.addEvent(event);

            }

        }
        switch (profilingMode) {
            case NAN_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                break;
            case INF_PANIC:
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case ANY_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case DISABLED:
            default:
                break;
        }

        if (Nd4j.getExecutioner().isVerbose()) {
            if (op.z() != null)
                log.info("Op name: {}; Z shapeInfo: {}; Z values: {}", op.opName(), op.z().shapeInfoJava(), firstX(op.z(), 10));
        }

        if(Nd4j.getEnvironment().isLogNDArrayEvents()) {
            INDArray z = op.z() != null ? op.z() : oc.getOutputArray(0);
            INDArray x = op.x() != null ? op.x() : oc.getInputArray(0);
            INDArray y = op.y() != null ? op.y() : oc.getInputArrays().size() >  1 ? oc.getInputArray(1) : null;
            if(x != null) {
                op.z().addEvent(NDArrayEvent.builder()
                        .parentDataAtEvent(NDArrayMetaData.fromArr(x))
                        .dataAtEvent(NDArrayMetaData.from(z))
                        .ndArrayEventType(NDArrayEventType.OP_OUTPUT)
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .build());
            }

            if(y != null) {
                op.z().addEvent(NDArrayEvent.builder()
                        .parentDataAtEvent(NDArrayMetaData.fromArr(y))
                        .dataAtEvent(NDArrayMetaData.from(z))
                        .ndArrayEventType(NDArrayEventType.OP_OUTPUT)
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .build());
            }

        }

    }

    @Override
    public Nd4jEventLog getNd4jEventLog() {
        return eventLog;
    }

    @Deprecated
    public void profilingHookOut(CustomOp op, OpContext oc, long timeStart) {
        if(Nd4j.getEnvironment().isLogNDArrayEvents()) {
            for(val arr : op.inputArguments()) {
                NDArrayEvent event = NDArrayEvent.builder()
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .parentDataAtEvent(NDArrayMetaData.fromArr(arr))
                        .dataAtEvent(NDArrayMetaData.from(arr))
                        .ndArrayEventType(NDArrayEventType.OP_INPUT)
                        .build();
                arr.addEvent(event);

            }

            for(val arr: op.outputArguments()) {
                for(val inputArr : op.inputArguments()) {
                    NDArrayEvent event = NDArrayEvent.builder()
                            .ndArrayEventType(NDArrayEventType.BEFORE_OP_OUTPUT)
                            .dataAtEvent(NDArrayMetaData.from(arr))
                            .parentDataAtEvent(NDArrayMetaData.fromArr(inputArr))
                            .stackTrace(Thread.currentThread().getStackTrace())
                            .build();
                    arr.addEvent(event);
                }

            }



        }
        switch (profilingMode) {
            case NAN_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                break;
            case INF_PANIC:
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case ANY_PANIC:
                OpExecutionerUtil.checkForNaN(op, oc);
                OpExecutionerUtil.checkForInf(op, oc);
                break;
            case DISABLED:
            default:
                break;
        }

        if(Nd4j.getEnvironment().isLogNDArrayEvents()) {
            for(val arr: op.outputArguments()) {
                for(val inputArr : op.inputArguments()) {
                    NDArrayEvent event = NDArrayEvent.builder()
                            .ndArrayEventType(NDArrayEventType.OP_OUTPUT)
                            .dataAtEvent(NDArrayMetaData.from(arr))
                            .parentDataAtEvent(NDArrayMetaData.fromArr(inputArr))
                            .stackTrace(Thread.currentThread().getStackTrace())
                            .build();
                    arr.addEvent(event);
                }

            }
        }

    }

    public static List<INDArray> inputArrsFromOp(Op op,OpContext opContext) {
        if(opContext != null && !opContext.getInputArrays().isEmpty()) {
            return opContext.getInputArrays();
        } else {
            if(op.x() != null && op.y() != null)
                return Arrays.asList(op.x(),op.y());
            else if(op.x() != null)
                return Collections.singletonList(op.x());
            else if(op.y() != null)
                return Collections.singletonList(op.y());
            else
                return Collections.emptyList();
        }
    }

    public static List<INDArray> outputArrsFromOp(Op op,OpContext opContext) {
        if(opContext != null && !opContext.getOutputArrays().isEmpty()) {
            return opContext.getOutputArrays();
        } else {
            if(op.z() != null)
                return Collections.singletonList(op.z());
            else if(op.y() != null)
                return Collections.singletonList(op.y());
            else if(op.x() != null)
                return Collections.singletonList(op.x());
            else
                return Collections.emptyList();
        }
    }

    public static List<INDArray> inputsFromOp(CustomOp customOp,OpContext opContext) {
        if(opContext != null && !opContext.getInputArrays().isEmpty()) {
            return opContext.getInputArrays();
        } else {
            return customOp.inputArguments();
        }
    }

    public static List<INDArray> outputsFromOp(CustomOp customOp,OpContext opContext) {
        if(opContext != null && !opContext.getOutputArrays().isEmpty()) {
            return opContext.getOutputArrays();
        } else {
            return customOp.outputArguments();
        }
    }

    public long profilingConfigurableHookIn(Op op, OpContext oc) {
        List<INDArray> inArgs = inputArrsFromOp(op,oc);
        List<INDArray> outArgs = outputArrsFromOp(op,oc);

        logOpArrayEventsIfNeccessary(op,inArgs ,outArgs, NDArrayEventType.BEFORE_OP_INPUT, NDArrayEventType.BEFORE_OP_OUTPUT);
        logOpArrayEventsIfNeccessary(op,inArgs ,outArgs, NDArrayEventType.OP_INPUT, NDArrayEventType.OP_OUTPUT);

        return System.nanoTime();
    }

    public long profilingConfigurableHookIn(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = inputsFromOp(op,oc);
        List<INDArray> outArgs = outputsFromOp(op,oc);
        Nd4j.getDeallocatorService().toggleDeallocationBlock(true);
        if(isDebug() && isVerbose()) {
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            String[] arg = differentialFunction.argNames();
            String[] output = differentialFunction.outputVariablesNames();
            log.info("About to execute op {} of type {} with inputs {} and outputs {}", differentialFunction.getOwnName(), op.opName(),
                    Arrays.toString(arg), Arrays.toString(differentialFunction.outputVariablesNames()));
        }

        logCustomOpArrayEventIfNeccessary(inArgs, outArgs,NDArrayEventType.BEFORE_OP_INPUT ,NDArrayEventType.BEFORE_OP_OUTPUT);
        logCustomOpArrayEventIfNeccessary(inArgs, outArgs,NDArrayEventType.OP_INPUT , NDArrayEventType.OP_OUTPUT);

        return System.nanoTime();
    }

    public long profilingConfigurableHookIn(Op op, DataBuffer... tadBuffers) {
        Nd4j.getDeallocatorService().toggleDeallocationBlock(true);

        List<INDArray> inputs = inputArrsFromOp(op,null);
        List<INDArray> outputs = outputArrsFromOp(op,null);
        logOpArrayEventsIfNeccessary(op,inputs,outputs, NDArrayEventType.BEFORE_OP_INPUT, NDArrayEventType.BEFORE_OP_OUTPUT);

        return System.nanoTime();

    }

    public void profilingConfigurableHookOut(Op op, OpContext oc, long timeStart) {
        Nd4j.getDeallocatorService().toggleDeallocationBlock(false);
        List<INDArray> inArgs = inputArrsFromOp(op,oc);
        List<INDArray> outArgs = outputArrsFromOp(op,oc);

        if (Nd4j.getExecutioner().isVerbose()) {
            if (op.z() != null)
                log.info("Op name: {}; Z shapeInfo: {}; Z values: {}", op.opName(), op.z().shapeInfoJava(), firstX(op.z(), 10));
        }

        if(profilerConfig.isCheckForNAN()) {
            OpExecutionerUtil.checkForNaN(op.z());
        } else if(profilerConfig.isCheckForINF()) {
            OpExecutionerUtil.checkForInf(op.z());
        }


        logOpArrayEventsIfNeccessary(op,inArgs ,outArgs, NDArrayEventType.OP_INPUT, NDArrayEventType.OP_OUTPUT);

    }

    private  void logOpArrayEventsIfNeccessary(Op op, List<INDArray> inArgs, List<INDArray> outArgs, NDArrayEventType eventType, NDArrayEventType outputEventType) {
        logArrays(inArgs, outArgs,eventType,outputEventType);
    }

    public void profilingConfigurableHookOut(CustomOp op, OpContext oc, long timeStart) {
        Nd4j.getDeallocatorService().toggleDeallocationBlock(true);
        List<INDArray> inArgs = inputsFromOp(op,oc);
        List<INDArray> outArgs = outputsFromOp(op,oc);
        logCustomOpArrayEventIfNeccessary(inArgs, outArgs,NDArrayEventType.OP_INPUT , NDArrayEventType.OP_OUTPUT);
        if(profilerConfig.isCheckForNAN()) {
            OpExecutionerUtil.checkForNaN(op,oc);
        } else if(profilerConfig.isCheckForINF()) {
            OpExecutionerUtil.checkForInf(op,oc);
        }
    }

    private void logCustomOpArrayEventIfNeccessary(List<INDArray> inArgs, List<INDArray> outArgs, NDArrayEventType inputEvenType, NDArrayEventType outputEventType) {
        logArrays(inArgs, outArgs,inputEvenType,outputEventType);
    }

    private static void logArrays(List<INDArray> inArgs, List<INDArray> outArgs, NDArrayEventType eventType, NDArrayEventType outputEventType) {
        List<NDArrayMetaData> inArgsMeta = new ArrayList<>();
        for (val arr: inArgs) {
            if(arr == null)
                continue;

            if (arr.wasClosed())
                throw new IllegalStateException("One of Input arguments was closed before call");

            if(Nd4j.getEnvironment().isLogNDArrayEvents() && !BaseNDArray.callingToString()) {
                NDArrayMetaData ndArrayMetaData = NDArrayMetaData.from(arr);
                NDArrayEvent event = NDArrayEvent.builder()
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .parentDataAtEvent(new NDArrayMetaData[]{ndArrayMetaData})
                        .dataAtEvent(ndArrayMetaData)
                        .ndArrayEventType(eventType)
                        .build();
                arr.addEvent(event);
                inArgsMeta.add(ndArrayMetaData);
            }

        }
        for (val arr: outArgs) {
            if(arr == null)
                continue;
            if (arr.wasClosed())
                throw new IllegalStateException("One of Output arguments was closed before call");

            if(Nd4j.getEnvironment().isLogNDArrayEvents() && !BaseNDArray.callingToString()) {
                NDArrayEvent event = NDArrayEvent.builder()
                        .stackTrace(Thread.currentThread().getStackTrace())
                        .parentDataAtEvent(inArgsMeta.toArray(new NDArrayMetaData[0]))
                        .dataAtEvent(NDArrayMetaData.from(arr))
                        .ndArrayEventType(outputEventType)
                        .build();
                arr.addEvent(event);


            }
        }
    }

    /**
     * Validate the data types
     * for the given operation
     * @param expectedType
     * @param op
     */
    public static void validateDataType(DataType expectedType, Op op) {
        if (op.x() != null && !Shape.isEmpty(op.x().shapeInfoJava()) && op.x().data().dataType() == DataType.COMPRESSED) {
            Nd4j.getCompressor().decompressi(op.x());
        }

        if (op.y() != null && !Shape.isEmpty(op.y().shapeInfoJava()) && op.y().data().dataType() == DataType.COMPRESSED) {
            Nd4j.getCompressor().decompressi(op.y());
        }

        if (op.z() != null && !Shape.isEmpty(op.z().shapeInfoJava()) && op.z().data().dataType() == DataType.COMPRESSED) {
            Nd4j.getCompressor().decompressi(op.z());
        }


        if (op.y() != null && !Shape.isEmpty(op.y().shapeInfoJava())
                && op.y().data().dataType() != expectedType) {
            throw new ND4JIllegalStateException("op.Y dataType is [" + op.y().data().dataType()
                    + "] instead of expected [" + expectedType + "] - x.shape = " + Arrays.toString(op.x().shape())
                    + (op.y() != null ? ", y.shape=" + Arrays.toString(op.y().shape()) : "")
                    + ", z.shape=" + Arrays.toString(op.z().shape()) + " - op: " + op.getClass().getName());

        }


        if (Nd4j.getExecutioner().isVerbose()) {
            log.info("Reporting [{}]", op.opName());
            if (op.x() != null)
                log.info("X shapeInfo: {}; X values: {}", op.x().shapeInfoJava(), firstX(op.x(), 10));

            if (op.y() != null)
                log.info("Y shapeInfo: {}; Y values: {}", op.y().shapeInfoJava(), firstX(op.y(), 10));
        }
    }

    protected static String firstX(INDArray array, int x) {
        val builder = new StringBuilder("[");
        val limit = (int) Math.min(x, array.length());
        for (int e = 0; e < limit; e++) {
            if(array.isS())
                builder.append(array.getString(e));
            else
                builder.append(array.getDouble(e));

            if (e < limit - 1)
                builder.append(", ");
        }
        builder.append("]");

        return builder.toString();
    }

    public static void validateDataType(DataType expectedType, Object op, INDArray... operands) {
        if (operands == null || operands.length == 0)
            return;

        int cnt = 0;
        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            if (operand.data().dataType() != expectedType) {
                throw new ND4JIllegalStateException("INDArray [" + cnt + "] dataType is [" + operand.data().dataType()
                        + "] instead of expected [" + expectedType + "]" + (op != null ? " op: " + op.getClass().getName() : ""));
            }
            cnt++;
        }
    }

    @Override
    public TADManager getTADManager() {
        throw new UnsupportedOperationException();
    }

    /**
     * This method return set of key/value and key/key/value objects, describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        Properties environment = new Properties();
        environment.put(Nd4jEnvironment.CPU_CORES_KEY, Runtime.getRuntime().availableProcessors());
        environment.put(Nd4jEnvironment.HOST_TOTAL_MEMORY_KEY, Runtime.getRuntime().maxMemory());
        environment.put(Nd4jEnvironment.OS_KEY, System.getProperty("os.name"));
        return environment;
    }

    @Override
    public void printEnvironmentInformation() {
        Properties env = getEnvironmentInformation();
        double memory = ((Long) env.get("memory.available")) / (double) 1024 / 1024 / 1024;
        String fm = String.format("%.1f", memory);
        log.info("Backend used: [{}]; OS: [{}]", env.get("backend"), env.get("os"));
        log.info("Cores: [{}]; Memory: [{}GB];", env.get("cores"), fm);
        log.info("Blas vendor: [{}]", env.get("blas.vendor"));
    }

    @Override
    public void push() {
        // no-op
    }

    @Override
    public void commit() {
        // no-op
    }

    @Override
    public Map<String, CustomOpDescriptor> getCustomOperations() {
        throw new UnsupportedOperationException();
    }


    public void execUdf(UserDefinedCustomOp userDefinedCustomOp) {
        userDefinedCustomOp.exec();
    }

    @Override
    public CustomOp execAndReturn(CustomOp op) {
        if(op instanceof UserDefinedCustomOp) {
            execUdf((UserDefinedCustomOp) op);
            return op;
        }

        exec(op);
        return op;
    }



    @Override
    public INDArray[] allocateOutputArrays(CustomOp op) {
        List<DataBuffer> shapes = calculateOutputShape(op);
        INDArray[] out = new INDArray[shapes.size()];
        for(int i = 0; i < shapes.size(); i++) {
            out[i] = Nd4j.createFromDescriptor(shapes.get(i));
        }
        return out;
    }

    @Override
    public int useCount(DataBuffer buffer) {
        return Nd4j.getNativeOps().dbUseCount(buffer.opaqueBuffer());
    }


    @Override
    public boolean isVerbose() {
        return verbose.get();
    }

    @Override
    public boolean isDebug() {
        return debug.get();
    }

    @Override
    public ExecutionerType type() {
        throw new UnsupportedOperationException();
    }


    /**
     * Get the information about the op in a String representation, for throwing more useful exceptions (mainly for debugging)
     * @param op
     * @param dimensions    Use optional here for 3 states: null = "not an exec(Op, int... dim) call". empty = "exec(Op, null)".
     *                     Otherwise present = "exec(Op,int[])" call
     * @return
     */
    public String opInfoString(Op op, Optional<long[]> dimensions){
        if(op == null)
            return "<NULL OP>";

        StringBuilder sb = new StringBuilder();
        sb.append("Class: ").append(op.getClass().getName()).append("; opNum: ").append(op.opNum())
                .append("; opName: ").append(op.opName());
        if(op instanceof DifferentialFunction) {
            sb.append("; opType: ").append(((DifferentialFunction)op).opType());
        }

        if(dimensions != null){
            sb.append("; dimensions: ");
            if(dimensions.isPresent()){
                sb.append(Arrays.toString(dimensions.get()));
            } else {
                sb.append("<null>");
            }
        }

        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();
        Object[] extraArgs = op.extraArgs();

        sb.append("\n");
        sb.append("x: ").append(arrayInfo(x)).append("; ");
        sb.append("y: ").append(arrayInfo(y)).append("; ");
        sb.append("z: ").append(arrayInfo(z)).append("; ");
        if(x == y && x != null)
            sb.append("(x == y)");
        if(x == z && x != null)
            sb.append("(x == z)");
        if(y == z && y != null)
            sb.append("(y == z)");
        sb.append("\n");
        sb.append("; extraArgs: ").append(Preconditions.formatArray(extraArgs));
        return sb.toString();
    }

    public String arrayInfo(INDArray arr) {
        if(arr == null)
            return "<null>";
        if(arr.isEmpty())
            return "(empty NDArray)";

        return arr.shapeInfoToString().replaceAll("\n","");
    }

    @Override
    public boolean isExperimentalMode() {
        return false;
    }

    @Override
    public OpContext buildContext() {
        throw new UnsupportedOperationException("OpContext is available only on native backends");
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty,boolean isView) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extras) {
        throw new UnsupportedOperationException();
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer createConstantBuffer(int[] values, DataType desiredType) {
        return createConstantBuffer(ArrayUtil.toLongArray(values), desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(float[] values, DataType desiredType) {
        return createConstantBuffer(ArrayUtil.toDoubles(values), desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType)  {
        throw new UnsupportedOperationException();
    }


    public INDArray getX(Op op, OpContext oc) {
        if( oc != null)
            return oc.getInputArray(0);
        return op.x();
    }

    public void setY(INDArray y, Op op, OpContext oc){
        if(oc != null)
            oc.setInputArray(1, y);
        else
            op.setY(y);
    }

    public INDArray getY(Op op, OpContext oc){
        if( oc != null )
            return oc.getInputArray(1);
        return op.y();
    }

    public void setZ(INDArray z, Op op, OpContext oc) {
        if(oc != null)
            oc.setOutputArray(0, z);
        else
            op.setZ(z);
    }

    public INDArray getZ(Op op, OpContext oc) {
        if( oc != null)
            return oc.getOutputArray(0);
        return op.z();
    }



    @Override
    public List<DataBuffer> calculateOutputShape(@NonNull CustomOp op) {
        try(OpContext ctx = buildContext()) {
            op.setupOpContextFromCustomOp(ctx);
            return calculateOutputShape(op, ctx);
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            clearOpContext();
        }
    }

    @Override
    public List<DataBuffer> calculateOutputShape(@NonNull CustomOp op, OpContext opContext) {
        val hash = op.opHash();
        val result = new ArrayList<DataBuffer>();

        OpaqueShapeList ptrptr;
        ptrptr = Nd4j.getNativeOps().calculateOutputShapes2(null, hash, opContext.contextPointer());

        if (Nd4j.getNativeOps().lastErrorCode() != 0) {
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native execution exec failed: ");
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(Nd4j.getNativeOps().lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
        if (ptrptr == null)
            throw new RuntimeException();



        if (Nd4j.getNativeOps().lastErrorCode() != 0) {
            StringBuilder errorMessage = new StringBuilder();
            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            errorMessage.append("Native execution exec failed: ");
            errorMessage.append(differentialFunction.debugInfo());
            errorMessage.append(Nd4j.getNativeOps().lastErrorMessage());
            throw new RuntimeException(errorMessage.toString());
        }
        if (ptrptr == null)
            throw new RuntimeException();

        for (int e = 0; e < Nd4j.getNativeOps().getShapeListSize(ptrptr); e++)
            result.add(getShapeFromPointer(op,opContext,new PagedPointer(Nd4j.getNativeOps().getShape(ptrptr, e)).asLongPointer()));

        if (log.isTraceEnabled()) {
            String[] arr = new String[result.size()];
            for (int i = 0; i < result.size(); i++) {
                arr[i] = result.get(i).toString();
            }

            DifferentialFunction differentialFunction = (DifferentialFunction) op;
            log.trace("Calculated output shapes for op of name {} and type {} - {}", differentialFunction.getOwnName(), op.getClass().getName(), Arrays.toString(arr));
        }
        return result;
    }

    protected DataBuffer getShapeFromPointer(CustomOp op,OpContext ctx,LongPointer ptr) {
        val rank = (int) ptr.get(0);
        int len = Shape.shapeInfoLength(rank);
        return Nd4j.createBuffer(ptr.capacity(len),Shape.shapeInfoLength(rank),DataType.INT64);
    }


    @Override
    public void enableDebugMode(boolean reallyEnable) {
        debug.set(reallyEnable);
        Nd4j.getNativeOps().enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        verbose.set(reallyEnable);
        Nd4j.getNativeOps().enableVerboseMode(reallyEnable);
    }


    @Override
    public void registerGraph(long id, Pointer graph) {
        Nd4j.getNativeOps().registerGraph(null, id, graph);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, @NonNull Map<String, INDArray> map, @NonNull Map<String, Integer> reverseMap) {

        val ptrBuffers = new PointerPointer(map.size());
        val ptrShapes = new PointerPointer(map.size());
        val ptrIndices = new IntPointer(map.size());

        int cnt = 0;
        val keySet = new ArrayList<String>(map.keySet());
        for (val key: keySet) {
            val array = map.get(key);

            ptrBuffers.put(cnt, array.data().addressPointer());
            ptrShapes.put(cnt, array.shapeInfoDataBuffer().addressPointer());
            ptrIndices.put(cnt, reverseMap.get(key));

            cnt++;
        }

        val newMap = new LinkedHashMap<String, INDArray>();

        OpaqueVariablesSet result = Nd4j.getNativeOps().executeStoredGraph(null, id, ptrBuffers, ptrShapes, ptrIndices, map.size());

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        OpStatus status = OpStatus.byNumber(Nd4j.getNativeOps().getVariablesSetStatus(result));

        if (status != OpStatus.ND4J_STATUS_OK)
            throw new ND4JIllegalStateException("Op execution failed: " + status);

        for (int e = 0; e < Nd4j.getNativeOps().getVariablesSetSize(result); e++) {
            OpaqueVariable var = Nd4j.getNativeOps().getVariable(result, e);
            int nodeId = Nd4j.getNativeOps().getVariableId(var);
            int index = Nd4j.getNativeOps().getVariableIndex(var);
            LongPointer shapeInfo = Nd4j.getNativeOps().getVariableShape(var);
            Pointer buffer = Nd4j.getNativeOps().getVariableBuffer(var);

            val rank = (int) shapeInfo.get(0);
            val jshape = new long[rank * 2 + 4];
            for (int i = 0; i < jshape.length; i++) {
                jshape[i] = shapeInfo.get(i);
            }

            val shapeOf = Shape.shapeOf(jshape);
            val stridesOf = Shape.stridesOf(jshape);
            val order = Shape.order(jshape);
            val array = Nd4j.create(shapeOf, stridesOf, 0, order);

            val perfX = PerformanceTracker.getInstance().helperStartTransaction();

            Pointer.memcpy(array.data().addressPointer(), buffer, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(array.dataType()));

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(array.dataType()), MemcpyDirection.HOST_TO_HOST);

            String nodeName = Nd4j.getNativeOps().getVariableName(var);
            newMap.put(nodeName, array);
        }

        // Nd4j.getNativeOps().deleteVariablesSet(result);

        return newMap;
    }

    @Override
    public void forgetGraph(long id) {
        Nd4j.getNativeOps().unregisterGraph(null, id);
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());
    }

    /**
     * This method allows to set desired number of elements per thread, for performance optimization purposes.
     * I.e. if array contains 2048 elements, and threshold is set to 1024, 2 threads will be used for given op execution.
     * <p>
     * Default value: 1024
     *
     * @param threshold
     */
    @Override
    public void setElementsThreshold(int threshold) {
        Nd4j.getNativeOps().setElementThreshold(threshold);
    }

    /**
     * This method allows to set desired number of sub-arrays per thread, for performance optimization purposes.
     * I.e. if matrix has shape of 64 x 128, and threshold is set to 8, each thread will be processing 8 sub-arrays (sure, if you have 8 core cpu).
     * If your cpu has, say, 4, cores, only 4 threads will be spawned, and each will process 16 sub-arrays
     * <p>
     * Default value: 8
     *
     * @param threshold
     */
    @Override
    public void setTadThreshold(int threshold) {
        Nd4j.getNativeOps().setTADThreshold(threshold);
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        val addr = ((LongIndexer) buffer.indexer()).get(index);
        val ptr = new PagedPointer(addr);
        return "";
    }
}
