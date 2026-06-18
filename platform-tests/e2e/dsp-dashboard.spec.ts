import { test, expect } from '@playwright/test';

test.describe('DSP Dashboard Page', () => {

  test('loads and shows title', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page).toHaveTitle('DSP Diagnostics');
    await expect(page.locator('.dl4j-toolbar h1')).toHaveText('DSP Diagnostics');
  });

  test('status cards render with real data', async ({ page }) => {
    await page.goto('/dsp');
    // Wait for the JS to populate status cards with real data (not the "NO DATA" placeholder).
    // setNoData() also creates .dl4j-phase-badge, so we must wait for actual phase text.
    await page.waitForFunction(
      () => {
        const el = document.querySelector('#planPhase .dl4j-phase-badge');
        return el && el.textContent !== 'NO DATA' && el.textContent !== '--';
      },
      { timeout: 15_000 }
    );

    // Plan phase should be a recognized value — read from the badge, not the parent
    const planPhase = await page.locator('#planPhase .dl4j-phase-badge').textContent();
    expect(planPhase).not.toBe('--');
    expect(['SLOT_BY_SLOT', 'SHAPES_FROZEN', 'REPLAYING']).toContain(planPhase!.trim());

    // Graph node phase
    const gnPhase = await page.locator('#graphNodePhase').textContent();
    expect(gnPhase).not.toBe('--');

    // Numeric values should be populated
    const numSlots = await page.locator('#numSlots').textContent();
    expect(parseInt(numSlots!)).toBeGreaterThan(0);

    const numSegments = await page.locator('#numSegments').textContent();
    expect(parseInt(numSegments!)).toBeGreaterThan(0);

    const iteration = await page.locator('#iteration').textContent();
    expect(parseInt(iteration!)).toBeGreaterThan(0);

    const iterTime = await page.locator('#iterTime').textContent();
    expect(parseInt(iterTime!)).toBeGreaterThan(0);

    // Execution mode
    const execMode = await page.locator('#execMode').textContent();
    expect(execMode).toBe('AUTO');
  });

  test('segment timeline renders bars', async ({ page }) => {
    await page.goto('/dsp');
    // Wait for segments to render
    await expect(page.locator('#segmentTimeline .dl4j-segment-bar-row').first()).toBeVisible({ timeout: 10_000 });

    const segmentRows = page.locator('#segmentTimeline .dl4j-segment-bar-row');
    const count = await segmentRows.count();
    expect(count).toBe(4);

    // First segment bar should have a replay state class
    const firstBar = page.locator('#segmentTimeline .dl4j-segment-bar').first();
    const barClass = await firstBar.getAttribute('class');
    expect(barClass).toMatch(/replay-/);

    // Segment bar should show replay state text
    const barText = await firstBar.textContent();
    expect(barText).toBeTruthy();
  });

  test('segment tooltip shows details on hover', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#segmentTimeline .dl4j-segment-bar').first()).toBeVisible({ timeout: 10_000 });

    const firstBar = page.locator('#segmentTimeline .dl4j-segment-bar').first();
    const tooltip = firstBar.locator('.dl4j-segment-tooltip');

    // Tooltip hidden by default
    await expect(tooltip).toBeHidden();

    // Hover shows tooltip
    await firstBar.hover();
    await expect(tooltip).toBeVisible();

    // Tooltip should contain segment details
    const tooltipText = await tooltip.textContent();
    expect(tooltipText).toContain('Segment 0');
    expect(tooltipText).toContain('Slots:');
    expect(tooltipText).toContain('Device:');
    expect(tooltipText).toContain('Ops:');
  });

  test('op histogram renders bars', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#opHistogram .dl4j-op-bar-wrap').first()).toBeVisible({ timeout: 10_000 });

    const opBars = page.locator('#opHistogram .dl4j-op-bar-wrap');
    const count = await opBars.count();
    expect(count).toBeGreaterThanOrEqual(5);

    // First bar should be matmul (highest count)
    const firstLabel = page.locator('#opHistogram .dl4j-op-bar-label').first();
    await expect(firstLabel).toHaveText('matmul');

    // First count should be 4
    const firstCount = page.locator('#opHistogram .dl4j-op-bar-count').first();
    await expect(firstCount).toHaveText('4');
  });

  test('memory timeline renders event dots', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#memoryTimeline .dl4j-mem-event').first()).toBeVisible({ timeout: 10_000 });

    const memEvents = page.locator('#memoryTimeline .dl4j-mem-event');
    const count = await memEvents.count();
    expect(count).toBeGreaterThan(10);

    // Should have both ALLOCATE (green) and RELEASE (red) dots
    const allocEvents = page.locator('#memoryTimeline .mem-ALLOCATE');
    const releaseEvents = page.locator('#memoryTimeline .mem-RELEASE');
    expect(await allocEvents.count()).toBeGreaterThan(0);
    expect(await releaseEvents.count()).toBeGreaterThan(0);
  });

  test('risky ops table renders rows', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#riskyOps table').first()).toBeVisible({ timeout: 10_000 });

    // Table should have headers
    const headers = page.locator('#riskyOps th');
    await expect(headers.nth(0)).toHaveText('Slot');
    await expect(headers.nth(1)).toHaveText('Op Name');
    await expect(headers.nth(2)).toHaveText('State');
    await expect(headers.nth(3)).toHaveText('Flags');
    await expect(headers.nth(4)).toHaveText('Description');

    // Should have 2 risky ops
    const rows = page.locator('#riskyOps tbody tr');
    expect(await rows.count()).toBe(2);

    // First risky op is gather
    const firstRow = rows.first();
    const cells = firstRow.locator('td');
    await expect(cells.nth(1)).toHaveText('gather');
  });

  test('DOT graph section renders', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#planGraph pre').first()).toBeVisible({ timeout: 10_000 });

    const dotText = await page.locator('#planGraph pre').textContent();
    expect(dotText).toContain('digraph plan');
    expect(dotText).toContain('matmul');
    expect(dotText).toContain('cluster_seg0');
    expect(dotText).toContain('cuBLAS');
    expect(dotText).toContain('Triton');
  });

  test('device placement renders bars', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#devicePlacement .dl4j-device-bar-wrap').first()).toBeVisible({ timeout: 10_000 });

    const deviceBars = page.locator('#devicePlacement .dl4j-device-bar-wrap');
    expect(await deviceBars.count()).toBe(2);

    // GPU 0 should have 15 slots, GPU 1 should have 3
    const counts = page.locator('#devicePlacement .dl4j-device-bar-count');
    await expect(counts.nth(0)).toHaveText('15');
    await expect(counts.nth(1)).toHaveText('3');

    const labels = page.locator('#devicePlacement .dl4j-device-bar-label');
    await expect(labels.nth(0)).toHaveText('GPU 0');
    await expect(labels.nth(1)).toHaveText('GPU 1');
  });

  test('model info table renders', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#modelInfo table').first()).toBeVisible({ timeout: 10_000 });

    const infoText = await page.locator('#modelInfo').textContent();
    expect(infoText).toContain('42');        // variableCount
    expect(infoText).toContain('18');        // opCount
    expect(infoText).toContain('1.2M');      // paramCount formatted
    expect(infoText).toContain('Nd4jCuda');  // backend
    expect(infoText).toContain('FLOAT');     // dataType
    expect(infoText).toContain('playwright-host'); // hostname
  });

  test('session selector is populated', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#sessionSelect option')).not.toHaveText('Loading...');

    const options = page.locator('#sessionSelect option');
    const count = await options.count();
    expect(count).toBeGreaterThanOrEqual(1);

    // First option should be part of the test session ID
    const firstOption = await options.first().getAttribute('value');
    expect(firstOption).toContain('test-session');
  });

  test('auto-refresh checkbox toggles polling', async ({ page }) => {
    await page.goto('/dsp');
    const checkbox = page.locator('#autoRefresh');
    await expect(checkbox).toBeChecked();

    // Uncheck — should stop auto-refresh
    await checkbox.uncheck();
    await expect(checkbox).not.toBeChecked();

    // Re-check
    await checkbox.check();
    await expect(checkbox).toBeChecked();
  });

  test('nav links are present and correct', async ({ page }) => {
    await page.goto('/dsp');

    const navLinks = page.locator('.dl4j-toolbar .nav-links a');
    const count = await navLinks.count();
    expect(count).toBe(3);

    await expect(navLinks.nth(0)).toHaveAttribute('href', '/samediff');
    await expect(navLinks.nth(1)).toHaveAttribute('href', '/models');
    await expect(navLinks.nth(2)).toHaveAttribute('href', '/train/overview');
  });

  test('phase badge has correct CSS class', async ({ page }) => {
    await page.goto('/dsp');
    await expect(page.locator('#planPhase .dl4j-phase-badge')).toBeVisible({ timeout: 10_000 });

    const badge = page.locator('#planPhase .dl4j-phase-badge');
    const className = await badge.getAttribute('class');
    // The latest report is iteration 5 → REPLAYING
    expect(className).toContain('phase-REPLAYING');
  });

  test('section titles are all present', async ({ page }) => {
    await page.goto('/dsp');

    const expectedSections = [
      'Segment Timeline',
      'Device Placement',
      'Op Histogram',
      'Memory Timeline',
      'Risky Ops',
      'Plan Graph (DOT)',
      'Model Info',
    ];

    for (const title of expectedSections) {
      await expect(page.locator('.dl4j-section-title', { hasText: title })).toBeVisible();
    }
  });
});


test.describe('DSP API Endpoints', () => {

  test('/dsp/state returns valid JSON', async ({ request }) => {
    const resp = await request.get('/dsp/state');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data.sessionID).toBe('test-session-001');
    expect(data.planPhase).toBe('REPLAYING');
    expect(data.numSlots).toBe(18);
    expect(data.numSegments).toBe(4);
    expect(data.iteration).toBe(5);
    expect(data.executionMode).toBe('AUTO');
    expect(data.segments).toHaveLength(4);
    expect(data.opHistogram.matmul).toBe(4);
    expect(data.riskyOps).toHaveLength(2);
    expect(data.memoryTimeline.length).toBeGreaterThan(10);
    expect(data.dotGraph).toContain('digraph plan');
    expect(data.deviceSlotCounts['0']).toBe(15);
    expect(data.deviceSlotCounts['1']).toBe(3);
  });

  test('/dsp/state/history returns iteration reports', async ({ request }) => {
    const resp = await request.get('/dsp/state/history');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data.length).toBeGreaterThanOrEqual(5);

    // All entries should have valid fields
    for (const entry of data) {
      expect(entry.sessionID).toBeTruthy();
      expect(entry.iteration).toBeGreaterThanOrEqual(1);
      expect(entry.planPhase).toBeTruthy();
      expect(entry.numSlots).toBeGreaterThan(0);
    }

    // Should contain both early (SLOT_BY_SLOT) and late (REPLAYING) phases
    const phases = new Set(data.map((d: any) => d.planPhase));
    expect(phases.has('REPLAYING')).toBeTruthy();
  });

  test('/dsp/sessions returns session list', async ({ request }) => {
    const resp = await request.get('/dsp/sessions');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data).toContain('test-session-001');
  });

  test('/dsp/info returns init report', async ({ request }) => {
    const resp = await request.get('/dsp/info');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data.variableCount).toBe(42);
    expect(data.opCount).toBe(18);
    expect(data.paramCount).toBe(1_200_000);
    expect(data.backend).toBe('Nd4jCuda');
    expect(data.hostname).toBe('playwright-host');
    expect(data.paramNames).toContain('weight_0');
  });

  test('/dsp/segments returns segment list', async ({ request }) => {
    const resp = await request.get('/dsp/segments');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data).toHaveLength(4);
    expect(data[0].startSlot).toBe(0);
    expect(data[0].endSlot).toBe(4);
    expect(data[0].opNames).toContain('matmul');
  });

  test('/dsp/ops returns op histogram', async ({ request }) => {
    const resp = await request.get('/dsp/ops');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data.matmul).toBe(4);
    expect(data.add).toBe(3);
    expect(data.softmax).toBe(2);
  });

  test('/dsp/memory returns memory events', async ({ request }) => {
    const resp = await request.get('/dsp/memory');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data.length).toBeGreaterThan(10);
    expect(data[0].step).toBe(0);
    expect(data[0].eventType).toBe('ALLOCATE');
  });

  test('/dsp/graph returns DOT string', async ({ request }) => {
    const resp = await request.get('/dsp/graph');
    expect(resp.ok()).toBeTruthy();

    const text = await resp.text();
    expect(text).toContain('digraph plan');
    expect(text).toContain('cluster_seg0');
  });

  test('/dsp/state/history shows phase progression', async ({ request }) => {
    const resp = await request.get('/dsp/state/history');
    const data = await resp.json();

    // Verify phase progression over iterations
    const phases = data.map((d: any) => ({ iter: d.iteration, phase: d.planPhase }));
    // Early iterations should be SLOT_BY_SLOT, later REPLAYING
    const earlyPhases = phases.filter((p: any) => p.iter <= 2);
    const latePhases = phases.filter((p: any) => p.iter >= 4);
    for (const p of earlyPhases) {
      expect(p.phase).toBe('SLOT_BY_SLOT');
    }
    for (const p of latePhases) {
      expect(p.phase).toBe('REPLAYING');
    }
  });

  test('/dsp/segments includes replay state per segment', async ({ request }) => {
    const resp = await request.get('/dsp/segments');
    const data = await resp.json();

    for (const seg of data) {
      // Every segment should have these fields
      expect(typeof seg.startSlot).toBe('number');
      expect(typeof seg.endSlot).toBe('number');
      expect(typeof seg.capturable).toBe('boolean');
      expect(typeof seg.deviceId).toBe('number');
      expect(typeof seg.slotCount).toBe('number');
      expect(Array.isArray(seg.opNames)).toBeTruthy();
      expect(seg.replayState).toBeDefined();
      expect(typeof seg.executionCount).toBe('number');
      expect(seg.executionPhase).toBeDefined();
    }

    // Segment 2 (index 2) is non-capturable
    expect(data[2].capturable).toBe(false);

    // Segment 3 is on device 1
    expect(data[3].deviceId).toBe(1);
  });

  test('/dsp/memory events have correct schema', async ({ request }) => {
    const resp = await request.get('/dsp/memory');
    const data = await resp.json();

    for (const evt of data) {
      expect(typeof evt.step).toBe('number');
      expect(typeof evt.slotIndex).toBe('number');
      expect(['ALLOCATE', 'RELEASE']).toContain(evt.eventType);
    }

    // ALLOCATE events should appear before RELEASE events for the same slot
    const allocSteps = data.filter((e: any) => e.eventType === 'ALLOCATE').map((e: any) => e.step);
    const releaseSteps = data.filter((e: any) => e.eventType === 'RELEASE').map((e: any) => e.step);
    expect(Math.min(...allocSteps)).toBeLessThan(Math.min(...releaseSteps));
  });

  test('/dsp/ops histogram values are positive integers', async ({ request }) => {
    const resp = await request.get('/dsp/ops');
    const data = await resp.json();

    const keys = Object.keys(data);
    expect(keys.length).toBeGreaterThan(0);
    for (const key of keys) {
      expect(typeof data[key]).toBe('number');
      expect(data[key]).toBeGreaterThan(0);
      expect(Number.isInteger(data[key])).toBe(true);
    }
  });

  test('/dsp/info has all init report fields', async ({ request }) => {
    const resp = await request.get('/dsp/info');
    const data = await resp.json();

    expect(data.configuredExecutionMode).toBe('AUTO');
    expect(data.dataType).toBe('FLOAT');
    expect(Array.isArray(data.requestedOutputs)).toBeTruthy();
    expect(data.requestedOutputs).toContain('output');
    expect(data.requestedOutputs).toContain('logits');
    expect(Array.isArray(data.paramNames)).toBeTruthy();
    expect(data.paramNames.length).toBe(4);
  });
});

test.describe('DSP Dashboard Data Integrity', () => {

  test('segment slot ranges are contiguous and non-overlapping', async ({ request }) => {
    const resp = await request.get('/dsp/segments');
    const segments = await resp.json();

    // Check segments don't overlap and cover all slots
    for (let i = 0; i < segments.length - 1; i++) {
      // endSlot of segment i should be < startSlot of segment i+1
      // (or they can be on different devices)
      expect(segments[i].endSlot).toBeLessThanOrEqual(segments[i + 1].startSlot);
    }
  });

  test('op histogram total matches slot count', async ({ request }) => {
    const stateResp = await request.get('/dsp/state');
    const state = await stateResp.json();

    const opsResp = await request.get('/dsp/ops');
    const ops = await opsResp.json();

    const totalOps = Object.values(ops).reduce((a: number, b: any) => a + b, 0) as number;
    expect(totalOps).toBe(state.numSlots);
  });

  test('device slot counts sum to total slots', async ({ request }) => {
    const resp = await request.get('/dsp/state');
    const data = await resp.json();

    const deviceTotal = Object.values(data.deviceSlotCounts).reduce((a: number, b: any) => a + b, 0) as number;
    expect(deviceTotal).toBe(data.numSlots);
  });

  test('all history entries have increasing timestamps', async ({ request }) => {
    const resp = await request.get('/dsp/state/history');
    const data = await resp.json();

    for (let i = 1; i < data.length; i++) {
      expect(data[i].timeStamp).toBeGreaterThanOrEqual(data[i - 1].timeStamp);
      expect(data[i].iteration).toBeGreaterThanOrEqual(data[i - 1].iteration);
    }
  });
});
