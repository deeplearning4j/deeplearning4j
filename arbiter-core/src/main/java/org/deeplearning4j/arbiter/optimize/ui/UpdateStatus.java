/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.ui;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class UpdateStatus {

    private long statusUpdateTime;
    private long settingsUpdateTime;
    private long resultsUpdateTime;

    public UpdateStatus(){

    }


    public long getStatusUpdateTime() {
        return this.statusUpdateTime;
    }

    public long getSettingsUpdateTime() {
        return this.settingsUpdateTime;
    }

    public long getResultsUpdateTime() {
        return this.resultsUpdateTime;
    }

    public void setStatusUpdateTime(long statusUpdateTime) {
        this.statusUpdateTime = statusUpdateTime;
    }

    public void setSettingsUpdateTime(long settingsUpdateTime) {
        this.settingsUpdateTime = settingsUpdateTime;
    }

    public void setResultsUpdateTime(long resultsUpdateTime) {
        this.resultsUpdateTime = resultsUpdateTime;
    }

    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof UpdateStatus)) return false;
        final UpdateStatus other = (UpdateStatus) o;
        if (!other.canEqual((Object) this)) return false;
        if (this.statusUpdateTime != other.statusUpdateTime) return false;
        if (this.settingsUpdateTime != other.settingsUpdateTime) return false;
        if (this.resultsUpdateTime != other.resultsUpdateTime) return false;
        return true;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final long $summaryLastUpdateTime = this.statusUpdateTime;
        result = result * PRIME + (int) ($summaryLastUpdateTime >>> 32 ^ $summaryLastUpdateTime);
        final long $settingsLastUpdateTime = this.settingsUpdateTime;
        result = result * PRIME + (int) ($settingsLastUpdateTime >>> 32 ^ $settingsLastUpdateTime);
        final long $resultsLastUpdateTime = this.resultsUpdateTime;
        result = result * PRIME + (int) ($resultsLastUpdateTime >>> 32 ^ $resultsLastUpdateTime);
        return result;
    }

    protected boolean canEqual(Object other) {
        return other instanceof UpdateStatus;
    }

    public String toString() {
        return "org.arbiter.optimize.ui.UpdateStatus(statusUpdateTime=" + this.statusUpdateTime + ", settingsUpdateTime=" + this.settingsUpdateTime + ", resultsUpdateTime=" + this.resultsUpdateTime + ")";
    }
}
