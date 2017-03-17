package org.nd4j.linalg.memory.provider;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceProvider;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

/**
 * @author raver119@gmail.com
 */
public class BasicWorkspaceProvider implements MemoryWorkspaceProvider {

    protected WorkspaceConfiguration defaultConfiguration;

    public BasicWorkspaceProvider() {
        this(WorkspaceConfiguration.builder().initialSize(0).maxSize(0).overallocationLimit(0.3).policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build());
    }

    public BasicWorkspaceProvider(@NonNull WorkspaceConfiguration defaultConfiguration) {
        this.defaultConfiguration = defaultConfiguration;
    }

    @Override
    public void setDefaultWorkspaceConfiguration(@NonNull WorkspaceConfiguration configuration) {
        this.defaultConfiguration = configuration;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration) {
        return new Nd4jWorkspace(configuration);
    }

    @Override
    public MemoryWorkspace createNewWorkspace() {
        return new Nd4jWorkspace(defaultConfiguration);
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread() {
        return null;
    }

    @Override
    public void setWorkspaceForCurrentThread(MemoryWorkspace workspace) {

    }

    @Override
    public void destroyWorkspace(MemoryWorkspace workspace) {

    }

    @Override
    public void destroyWorkspace() {

    }
}
