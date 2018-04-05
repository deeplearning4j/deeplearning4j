package org.deeplearning4j.datasets.iterator.callbacks;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This callback migrates incoming datasets in round-robin manner, to ensure TDA for ParallelWrapper
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class InterleavedDataSetCallback implements DataSetCallback {
    private List<MemoryWorkspace> workspaces = new ArrayList<>();
    private int bufferSize;
    private int numWorkspaces;

    private boolean isInitialized = false;

    private AtomicLong counterInput = new AtomicLong(0);

    public InterleavedDataSetCallback(int bufferSize) {
        this.bufferSize = bufferSize;
    }

    protected void initializeWorkspaces(long size) {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(size)
                        .overallocationLimit(bufferSize).policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.EXTERNAL)
                        .policyLearning(LearningPolicy.NONE).build();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        int cDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        for (int i = 0; i < numDevices; i++) {
            Nd4j.getAffinityManager().unsafeSetDevice(i);
            workspaces.add(Nd4j.getWorkspaceManager().createNewWorkspace(configuration, "IDSC-" + i, i));
        }

        Nd4j.getAffinityManager().unsafeSetDevice(cDevice);
        numWorkspaces = numDevices;
        isInitialized = true;
    }

    @Override
    public void call(DataSet dataSet) {
        if (!isInitialized)
            initializeWorkspaces(dataSet.getMemoryFootprint());

        Nd4j.getExecutioner().commit();

        int currIdx = (int) (counterInput.getAndIncrement() % numWorkspaces);
        MemoryWorkspace currWs = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(workspaces.get(currIdx));

        dataSet.migrate();

        Nd4j.getMemoryManager().setCurrentWorkspace(currWs);
    }

    @Override
    public void call(MultiDataSet multiDataSet) {
        if (!isInitialized)
            initializeWorkspaces(multiDataSet.getMemoryFootprint());

        Nd4j.getExecutioner().commit();

        int currIdx = (int) (counterInput.getAndIncrement() % numWorkspaces);
        MemoryWorkspace currWs = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(workspaces.get(currIdx));

        multiDataSet.migrate();

        Nd4j.getMemoryManager().setCurrentWorkspace(currWs);
    }

    @Override
    public void reset() {
        counterInput.set(0);
    }
}
