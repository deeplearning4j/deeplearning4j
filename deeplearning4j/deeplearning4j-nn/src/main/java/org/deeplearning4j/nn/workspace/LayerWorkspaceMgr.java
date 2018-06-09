package org.deeplearning4j.nn.workspace;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.workspace.BaseWorkspaceMgr;
import org.nd4j.linalg.workspace.WorkspaceMgr;

import java.util.*;

/**
 * {@link WorkspaceMgr} for DL4J layers.
 * Used to flexibly specify which workspaces a given array type (defined by {@link ArrayType}) should be placed in
 *
 * @author Alex Black
 */
public class LayerWorkspaceMgr extends BaseWorkspaceMgr<ArrayType> {
    public static String CUDNN_WORKSPACE_KEY = "CUDNN_WORKSPACE";

    private static LayerWorkspaceMgr NO_WS_IMMUTABLE;
    static{
        Set<ArrayType> all = new HashSet<>();
        Collections.addAll(all, ArrayType.values());
        NO_WS_IMMUTABLE = new LayerWorkspaceMgr(
                all, Collections.<ArrayType, WorkspaceConfiguration>emptyMap(), Collections.<ArrayType, String>emptyMap());
    }

    protected Set<String> noLeverageOverride;

    @Setter @Getter
    protected Map<String,Pointer> helperWorkspacePointers;

    private LayerWorkspaceMgr(){

    }

    public LayerWorkspaceMgr(Set<ArrayType> scopeOutOfWs, Map<ArrayType, WorkspaceConfiguration> configMap,
                             Map<ArrayType, String> workspaceNames){
        super(scopeOutOfWs, configMap, workspaceNames);
        if(configMap != null){
            Preconditions.checkArgument(configMap.keySet().equals(workspaceNames.keySet()),
                    "Keys for config may and workspace names must match");
        }
    }

    public void setNoLeverageOverride(String wsName){
        if(noLeverageOverride == null){
            noLeverageOverride = new HashSet<>();
        }
        noLeverageOverride.add(wsName);
    }

    @Override
    public INDArray leverageTo(ArrayType arrayType, INDArray array){
        if(noLeverageOverride != null && array.isAttached() && noLeverageOverride.contains(array.data().getParentWorkspace().getId())){
            return array;
        }
        return super.leverageTo(arrayType, array);
    }

    @Override
    public INDArray validateArrayLocation(@NonNull ArrayType arrayType, @NonNull INDArray array, boolean migrateIfInvalid, boolean exceptionIfDetached) {
        if(noLeverageOverride != null && array.isAttached() && noLeverageOverride.contains(array.data().getParentWorkspace().getId())){
            return array;   //OK - leverage override
        }
        return super.validateArrayLocation(arrayType, array, migrateIfInvalid, exceptionIfDetached);
    }

    public <T extends Pointer> T getHelperWorkspace(String key){
//        if(helperWorkspacePointers == null){
//            return null;
//        }
        return (T)helperWorkspacePointers.get(key);
    }

    public void setHelperWorkspace(@NonNull String key, Pointer value){
//        if(helperWorkspacePointers == null){
//            helperWorkspacePointers = new HashMap<>();
//        }
        helperWorkspacePointers.put(key, value);
    }

    public static Builder builder(){
        return new Builder();
    }

    public static LayerWorkspaceMgr noWorkspaces(Map<String,Pointer> helperWorkspacePointers){
        LayerWorkspaceMgr wsm = noWorkspaces();
        wsm.setHelperWorkspacePointers(helperWorkspacePointers);
        return wsm;
    }

    public static LayerWorkspaceMgr noWorkspaces(){
        return builder().defaultNoWorkspace().build();
    }

    public static LayerWorkspaceMgr noWorkspacesImmutable(){
        return NO_WS_IMMUTABLE;
    }

    public static class Builder {

        private LayerWorkspaceMgr mgr;

        public Builder(){
            mgr = new LayerWorkspaceMgr();
        }

        /**
         * Set the default to be scoped out for all array types.
         * NOTE: Will not override the configuration for any array types that have already been configured
         * @return Builder
         */
        public Builder defaultNoWorkspace(){
            for(ArrayType t : ArrayType.values()){
                if(!mgr.configMap.containsKey(t)){
                    mgr.setScopedOutFor(t);
                }
            }
            return this;
        }

        /**
         * Specify that no workspace should be used for array of the specified type - i.e., these arrays should all
         * be scoped out.
         *
         * @param type Array type to set scoped out for
         * @return Builder
         */
        public Builder noWorkspaceFor(ArrayType type){
            mgr.setScopedOutFor(type);
            return this;
        }

        /**
         * Set the default workspace for all array types to the specified workspace name/configuration
         * NOTE: This will NOT override any settings previously set.
         *
         * @param workspaceName Name of the workspace to use for all (not set) arrray types
         * @param configuration Configuration to use for all (not set) arrray types
         * @return Builder
         */
        public Builder defaultWorkspace(String workspaceName, WorkspaceConfiguration configuration){
            for(ArrayType t : ArrayType.values()){
                if(!mgr.configMap.containsKey(t) && !mgr.isScopedOut(t)){
                    with(t, workspaceName, configuration);
                }
            }
            return this;
        }

        /**
         * Configure the workspace (name, configuration) for the specified array type
         *
         * @param type          Array type
         * @param workspaceName Workspace name for the specified array type
         * @param configuration Configuration for the specified array type
         * @return Builder
         */
        public Builder with(ArrayType type, String workspaceName, WorkspaceConfiguration configuration){
            mgr.setConfiguration(type, configuration);
            mgr.setWorkspaceName(type, workspaceName);
            return this;
        }

        public LayerWorkspaceMgr build(){
            return mgr;
        }

    }
    
}
