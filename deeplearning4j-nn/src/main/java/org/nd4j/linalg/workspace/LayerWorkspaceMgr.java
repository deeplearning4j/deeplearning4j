package org.nd4j.linalg.workspace;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

public class LayerWorkspaceMgr extends BaseWorkspaceMgr<ArrayType> {
    
    public LayerWorkspaceMgr(){

    }

    public static Builder builder(){
        return new Builder();
    }

    public static LayerWorkspaceMgr noWorkspaces(){
        return builder().defaultNoWorkspace().build();
    }

    public static class Builder {

        private LayerWorkspaceMgr mgr;

        public Builder(){
            mgr = new LayerWorkspaceMgr();
        }

        public Builder defaultNoWorkspace(){
            for(ArrayType t : ArrayType.values()){
                if(!mgr.configMap.containsKey(t)){
                    mgr.setScopedOutFor(t);
                }
            }
            return this;
        }

        public Builder noWorkspaceFor(ArrayType type){
            mgr.setScopedOutFor(type);
            return this;
        }

        public Builder defaultWorkspace(String workspaceName, WorkspaceConfiguration configuration){
            for(ArrayType t : ArrayType.values()){
                if(!mgr.configMap.containsKey(t)){
                    with(t, workspaceName, configuration);
                }
            }
            return this;
        }

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
