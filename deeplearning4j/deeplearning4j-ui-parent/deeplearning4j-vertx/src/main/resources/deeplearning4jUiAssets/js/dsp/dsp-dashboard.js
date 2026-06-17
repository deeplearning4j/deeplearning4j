/*
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

var refreshInterval = null;
var currentSession = null;

$(document).ready(function() {
    loadSessions();
    startAutoRefresh();
});

function startAutoRefresh() {
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(function() {
        if ($('#autoRefresh').is(':checked')) {
            refreshAll();
        }
    }, 2000);
}

function toggleAutoRefresh() {
    if (!$('#autoRefresh').is(':checked') && refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    } else {
        startAutoRefresh();
    }
}

function onSessionChange() {
    currentSession = $('#sessionSelect').val();
    refreshAll();
}

function sessionParam() {
    return currentSession ? '?session=' + encodeURIComponent(currentSession) : '';
}

function loadSessions() {
    $.getJSON('/dsp/sessions', function(sessions) {
        var sel = $('#sessionSelect');
        sel.empty();
        if (sessions.length === 0) {
            sel.append('<option value="">No sessions</option>');
        } else {
            for (var i = 0; i < sessions.length; i++) {
                var sid = sessions[i];
                var label = sid.length > 12 ? sid.substring(0, 12) + '...' : sid;
                sel.append('<option value="' + sid + '">' + label + '</option>');
            }
            currentSession = sessions[0];
        }
        refreshAll();
    }).fail(function() {
        refreshAll();
    });
}

function refreshAll() {
    loadState();
    loadInfo();
}

function loadState() {
    $.getJSON('/dsp/state' + sessionParam(), function(data) {
        if (data.error) {
            setNoData();
            return;
        }
        renderState(data);
    }).fail(function() {
        setNoData();
    });
}

function loadInfo() {
    $.getJSON('/dsp/info' + sessionParam(), function(data) {
        if (!data.error) {
            renderInfo(data);
        }
    });
}

function setNoData() {
    $('#planPhase').html('<span class="dl4j-phase-badge" style="background:var(--text-muted)">NO DATA</span>');
    $('#graphNodePhase').text('--');
    $('#execMode').text('--');
    $('#numSlots').text('--');
    $('#numSegments').text('--');
    $('#iteration').text('--');
    $('#iterTime').text('--');
    $('#unfrozenOps').text('--');
}

function renderState(d) {
    // Status cards
    var phaseClass = 'phase-' + (d.planPhase || 'UNKNOWN');
    $('#planPhase').html('<span class="dl4j-phase-badge ' + phaseClass + '">' + (d.planPhase || '--') + '</span>');

    var gnPhaseClass = 'phase-' + (d.graphNodePhase || 'UNKNOWN');
    $('#graphNodePhase').html('<span class="dl4j-phase-badge ' + gnPhaseClass + '">' + (d.graphNodePhase || '--') + '</span>');

    $('#execMode').text(d.executionMode || '--');
    $('#numSlots').text(d.numSlots != null ? d.numSlots : '--');
    $('#numSegments').text(d.numSegments != null ? d.numSegments : '--');
    $('#iteration').text(d.iteration != null ? d.iteration : '--');
    $('#iterTime').text(d.iterationTimeMs != null ? d.iterationTimeMs : '--');
    $('#unfrozenOps').text(d.unfrozenOpCount != null ? d.unfrozenOpCount : '--');

    // Segments
    renderSegments(d.segments);

    // Op histogram
    renderOpHistogram(d.opHistogram);

    // Memory timeline
    renderMemoryTimeline(d.memoryTimeline);

    // Risky ops
    renderRiskyOps(d.riskyOps);

    // DOT graph
    renderDotGraph(d.dotGraph);

    // Device placement
    renderDevicePlacement(d.deviceSlotCounts);
}

function renderSegments(segments) {
    var container = $('#segmentTimeline');
    if (!segments || segments.length === 0) {
        container.html('<div class="dl4j-no-data">No segment data</div>');
        return;
    }

    var maxSlot = 0;
    for (var i = 0; i < segments.length; i++) {
        if (segments[i].endSlot > maxSlot) maxSlot = segments[i].endSlot;
    }
    if (maxSlot === 0) maxSlot = 1;

    var html = '';
    for (var i = 0; i < segments.length; i++) {
        var seg = segments[i];
        var widthPct = Math.max(3, ((seg.slotCount || (seg.endSlot - seg.startSlot + 1)) / maxSlot) * 100);
        var replayClass = 'replay-' + (seg.replayState || 'UNKNOWN');
        var execClass = 'exec-' + (seg.executionPhase || 'UNKNOWN');
        var ops = (seg.opNames || []).join(', ');
        if (ops.length > 60) ops = ops.substring(0, 60) + '...';

        html += '<div class="dl4j-segment-bar-row">';
        html += '<span class="dl4j-segment-label">Seg ' + i + ' [' + seg.startSlot + '-' + seg.endSlot + ']</span>';
        html += '<div class="dl4j-segment-bar ' + replayClass + '" style="width:' + widthPct + '%">';
        html += (seg.replayState || '?');
        if (seg.executionPhase && seg.executionPhase !== 'UNKNOWN') {
            html += '<span class="dl4j-exec-badge ' + execClass + '">' + seg.executionPhase + '</span>';
        }
        html += '<div class="dl4j-segment-tooltip">';
        html += '<b>Segment ' + i + '</b><br>';
        html += 'Slots: ' + seg.startSlot + ' - ' + seg.endSlot + ' (' + (seg.slotCount || '?') + ' slots)<br>';
        html += 'Capturable: ' + (seg.capturable ? 'Yes' : 'No') + '<br>';
        html += 'Device: ' + seg.deviceId + '<br>';
        html += 'Backend: ' + (seg.backendName || 'N/A') + '<br>';
        html += 'Replay: ' + (seg.replayState || 'N/A') + ' (count: ' + (seg.replayCount || 0) + ')<br>';
        html += 'Executions: ' + (seg.executionCount || 0) + '<br>';
        html += 'Phase: ' + (seg.executionPhase || 'N/A') + '<br>';
        if (seg.compilationFailed) html += '<span style="color:var(--error)">Compilation FAILED</span><br>';
        html += 'Ops: ' + (ops || 'none') + '<br>';
        html += '</div>';
        html += '</div>';
        html += '</div>';
    }
    container.html(html);
}

function renderOpHistogram(histogram) {
    var container = $('#opHistogram');
    if (!histogram || Object.keys(histogram).length === 0) {
        container.html('<div class="dl4j-no-data">No op data</div>');
        return;
    }

    var entries = [];
    for (var op in histogram) {
        entries.push({name: op, count: histogram[op]});
    }
    entries.sort(function(a, b) { return b.count - a.count; });

    var maxCount = entries[0].count;
    var displayEntries = entries.slice(0, 20);

    var html = '';
    for (var i = 0; i < displayEntries.length; i++) {
        var e = displayEntries[i];
        var heightPct = (e.count / maxCount) * 100;
        html += '<div class="dl4j-op-bar-wrap">';
        html += '<span class="dl4j-op-bar-count">' + e.count + '</span>';
        html += '<div class="dl4j-op-bar" style="height:' + heightPct + '%"></div>';
        html += '<span class="dl4j-op-bar-label" title="' + e.name + '">' + e.name + '</span>';
        html += '</div>';
    }
    container.html(html);
}

function renderMemoryTimeline(events) {
    var container = $('#memoryTimeline');
    if (!events || events.length === 0) {
        container.html('<div class="dl4j-no-data">No memory data</div>');
        return;
    }

    var maxStep = 0;
    var maxSlot = 0;
    for (var i = 0; i < events.length; i++) {
        if (events[i].step > maxStep) maxStep = events[i].step;
        if (events[i].slotIndex > maxSlot) maxSlot = events[i].slotIndex;
    }
    if (maxStep === 0) maxStep = 1;
    if (maxSlot === 0) maxSlot = 1;

    var containerWidth = container.width() || 600;
    var containerHeight = 100;

    var html = '';
    for (var i = 0; i < events.length; i++) {
        var ev = events[i];
        var left = (ev.step / maxStep) * (containerWidth - 10);
        var top = (ev.slotIndex / maxSlot) * (containerHeight - 10);
        var cls = 'mem-' + (ev.eventType || 'ALLOCATE');
        html += '<div class="dl4j-mem-event ' + cls + '" style="left:' + left + 'px;top:' + top + 'px" title="Step ' + ev.step + ', Slot ' + ev.slotIndex + ': ' + ev.eventType + '"></div>';
    }
    container.html(html);
}

function renderRiskyOps(riskyOps) {
    var container = $('#riskyOps');
    if (!riskyOps || riskyOps.length === 0) {
        container.html('<div class="dl4j-no-data">No risky ops detected</div>');
        return;
    }

    var html = '<table class="dl4j-table"><thead><tr><th>Slot</th><th>Op Name</th><th>State</th><th>Flags</th><th>Description</th></tr></thead><tbody>';
    for (var i = 0; i < riskyOps.length; i++) {
        var op = riskyOps[i];
        html += '<tr>';
        html += '<td>' + op.index + '</td>';
        html += '<td>' + (op.opName || '--') + '</td>';
        html += '<td>' + (op.state || '--') + '</td>';
        html += '<td>' + (op.flags || 0) + '</td>';
        html += '<td>' + (op.flagDescription || '--') + '</td>';
        html += '</tr>';
    }
    html += '</tbody></table>';
    container.html(html);
}

function renderDotGraph(dotGraph) {
    var container = $('#planGraph');
    if (!dotGraph) {
        container.html('<div class="dl4j-no-data">No graph data</div>');
        return;
    }
    container.html('<pre>' + escapeHtml(dotGraph) + '</pre>');
}

function renderDevicePlacement(deviceSlotCounts) {
    var container = $('#devicePlacement');
    if (!deviceSlotCounts || Object.keys(deviceSlotCounts).length === 0) {
        container.html('<div class="dl4j-no-data">No device data</div>');
        return;
    }

    var maxCount = 0;
    for (var dev in deviceSlotCounts) {
        if (deviceSlotCounts[dev] > maxCount) maxCount = deviceSlotCounts[dev];
    }
    if (maxCount === 0) maxCount = 1;

    var html = '';
    var idx = 0;
    for (var dev in deviceSlotCounts) {
        var count = deviceSlotCounts[dev];
        var heightPx = Math.max(4, (count / maxCount) * 50);
        html += '<div class="dl4j-device-bar-wrap">';
        html += '<span class="dl4j-device-bar-count">' + count + '</span>';
        html += '<div class="dl4j-device-bar" style="height:' + heightPx + 'px;background:var(--chart-' + ((idx % 6) + 1) + ')"></div>';
        html += '<span class="dl4j-device-bar-label">GPU ' + dev + '</span>';
        html += '</div>';
        idx++;
    }
    container.html(html);
}

function renderInfo(info) {
    var container = $('#modelInfo');
    var html = '<table class="dl4j-table">';
    html += '<tr><td style="width:180px">Variables</td><td>' + (info.variableCount || 0) + '</td></tr>';
    html += '<tr><td>Ops</td><td>' + (info.opCount || 0) + '</td></tr>';
    html += '<tr><td>Parameters</td><td>' + formatNumber(info.paramCount || 0) + '</td></tr>';
    html += '<tr><td>Backend</td><td>' + (info.backend || '--') + '</td></tr>';
    html += '<tr><td>Data Type</td><td>' + (info.dataType || '--') + '</td></tr>';
    html += '<tr><td>Hostname</td><td>' + (info.hostname || '--') + '</td></tr>';
    if (info.configuredExecutionMode) {
        html += '<tr><td>Configured Mode</td><td>' + info.configuredExecutionMode + '</td></tr>';
    }
    if (info.paramNames && info.paramNames.length > 0) {
        var names = info.paramNames.join(', ');
        if (names.length > 200) names = names.substring(0, 200) + '...';
        html += '<tr><td>Param Names</td><td style="font-size:11px">' + escapeHtml(names) + '</td></tr>';
    }
    html += '</table>';
    container.html(html);
}

function formatNumber(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
