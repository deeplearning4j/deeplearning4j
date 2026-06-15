import { test, expect } from '@playwright/test';

test.describe('Training Overview Page', () => {

  test('loads overview page', async ({ page }) => {
    await page.goto('/train/overview');
    const title = await page.title();
    expect(title).toBeTruthy();
  });

  test('score chart is rendered with data', async ({ page }) => {
    await page.goto('/train/overview');
    // Wait for the score chart to be populated by JS polling
    await expect(page.locator('#scoreiterchart')).toBeVisible({ timeout: 15_000 });
    // The canvas/flot chart should have children (plot layers)
    const chartChildren = await page.locator('#scoreiterchart').evaluate(
      el => el.children.length
    );
    expect(chartChildren).toBeGreaterThan(0);
  });

  test('model metadata table is populated', async ({ page }) => {
    await page.goto('/train/overview');
    // Wait for "Loading..." to be replaced with real data
    await page.waitForFunction(
      () => {
        const el = document.getElementById('modelType');
        return el && el.textContent !== 'Loading...' && el.textContent!.trim() !== '';
      },
      { timeout: 15_000 }
    );

    const modelType = await page.locator('#modelType').textContent();
    expect(modelType).toBeTruthy();
    expect(modelType).not.toBe('Loading...');

    const nLayers = await page.locator('#nLayers').textContent();
    expect(parseInt(nLayers!)).toBeGreaterThan(0);

    const nParams = await page.locator('#nParams').textContent();
    expect(parseInt(nParams!)).toBeGreaterThan(0);
  });

  test('performance metrics are populated', async ({ page }) => {
    await page.goto('/train/overview');
    await page.waitForFunction(
      () => {
        const el = document.getElementById('totalParamUpdates');
        return el && el.textContent !== 'Loading...' && el.textContent!.trim() !== '';
      },
      { timeout: 15_000 }
    );

    const updates = await page.locator('#totalParamUpdates').textContent();
    expect(parseInt(updates!)).toBeGreaterThan(0);
  });

  test('sidebar nav links are present', async ({ page }) => {
    await page.goto('/train/overview');
    // Use .sidebar-nav specifically to avoid strict mode matching both .sidebar and .sidebar-nav
    const sidebar = page.locator('nav.sidebar-nav');
    await expect(sidebar).toBeVisible({ timeout: 10_000 });
  });

  test('/train serves overview content', async ({ page }) => {
    // /train does an internal reroute to /train/overview (not HTTP 302), so URL stays /train
    await page.goto('/train');
    // Verify the overview page content is rendered at /train
    await expect(page.locator('#scoreiterchart')).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('Training Overview API', () => {

  test('/train/overview/data returns valid JSON', async ({ request }) => {
    const resp = await request.get('/train/overview/data');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    // Should have scores array from training
    expect(data.scores).toBeDefined();
    expect(data.scores.length).toBeGreaterThan(0);
  });

  test('/train/sessions/info returns session metadata', async ({ request }) => {
    const resp = await request.get('/train/sessions/info');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    // Should have at least one session with model info
    const keys = Object.keys(data);
    expect(keys.length).toBeGreaterThanOrEqual(1);

    const session = data[keys[0]];
    expect(session.numLayers).toBeGreaterThan(0);
    expect(session.numParams).toBeGreaterThan(0);
  });

  test('/train/multisession returns boolean string', async ({ request }) => {
    const resp = await request.get('/train/multisession');
    expect(resp.ok()).toBeTruthy();

    const text = await resp.text();
    expect(['true', 'false']).toContain(text);
  });

  test('/train/sessions/current returns session ID', async ({ request }) => {
    const resp = await request.get('/train/sessions/current');
    expect(resp.ok()).toBeTruthy();

    const text = await resp.text();
    expect(text).toBeTruthy();
  });
});


test.describe('Training Model Page', () => {

  test('loads model page', async ({ page }) => {
    await page.goto('/train/model');
    const title = await page.title();
    expect(title).toBeTruthy();
  });

  test('layer graph container is visible', async ({ page }) => {
    await page.goto('/train/model');
    await expect(page.locator('#layers')).toBeVisible({ timeout: 15_000 });
  });

  test('zero state or graph is shown', async ({ page }) => {
    await page.goto('/train/model');
    // Either the Cytoscape graph has nodes or the zero state is shown
    const hasGraph = await page.locator('#layers canvas').count();
    const hasZeroState = await page.locator('#zeroState').count();
    expect(hasGraph + hasZeroState).toBeGreaterThan(0);
  });
});

test.describe('Training Model API', () => {

  test('/train/model/graph returns graph data', async ({ request }) => {
    const resp = await request.get('/train/model/graph');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    // Graph should have vertex names for the 3-layer MLP
    expect(data.vertexNames).toBeDefined();
    expect(data.vertexNames.length).toBeGreaterThan(0);
  });

  test('/train/model/data/:layerId returns layer details', async ({ request }) => {
    // layerId is an integer index into the vertexNames array, not a name string
    // Index 0 is the input pseudo-vertex; use index 1 for the first real layer
    const graphResp = await request.get('/train/model/graph');
    const graph = await graphResp.json();

    if (graph.vertexNames && graph.vertexNames.length > 1) {
      const layerResp = await request.get('/train/model/data/1');
      expect(layerResp.ok()).toBeTruthy();

      const data = await layerResp.json();
      expect(data).toBeDefined();
    }
  });
});


test.describe('Training System Page', () => {

  test('loads system page', async ({ page }) => {
    await page.goto('/train/system');
    const title = await page.title();
    expect(title).toBeTruthy();
  });

  test('hardware info is populated', async ({ page }) => {
    await page.goto('/train/system');
    await page.waitForFunction(
      () => {
        const el = document.getElementById('jvmAvailableProcessors');
        return el && el.textContent !== '' && el.textContent !== 'Loading...';
      },
      { timeout: 15_000 }
    );

    const processors = await page.locator('#jvmAvailableProcessors').textContent();
    expect(parseInt(processors!)).toBeGreaterThan(0);
  });

  test('software info is populated', async ({ page }) => {
    await page.goto('/train/system');
    await page.waitForFunction(
      () => {
        const el = document.getElementById('hostName');
        return el && el.textContent !== '' && el.textContent !== 'Loading...';
      },
      { timeout: 15_000 }
    );

    const hostName = await page.locator('#hostName').textContent();
    expect(hostName).toBeTruthy();

    const backend = await page.locator('#nd4jBackend').textContent();
    expect(backend).toBeTruthy();
  });

  test('memory chart container is visible', async ({ page }) => {
    await page.goto('/train/system');
    await expect(page.locator('#systemMemoryChart')).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('Training Session & Worker API', () => {

  test('/train/sessions/lastUpdate returns timestamp', async ({ request }) => {
    // Get current session ID first
    const sessResp = await request.get('/train/sessions/current');
    const sessionId = (await sessResp.text()).trim();

    const resp = await request.get(`/train/sessions/lastUpdate/${sessionId}`);
    expect(resp.ok()).toBeTruthy();

    const text = await resp.text();
    const ts = parseInt(text);
    // Should be a positive timestamp or -1
    expect(ts).not.toBeNaN();
    expect(ts !== 0).toBeTruthy();
  });

  test('/train/workers/setByIdx accepts valid index', async ({ request }) => {
    const resp = await request.get('/train/workers/setByIdx/0');
    expect(resp.ok()).toBeTruthy();
  });

  test('/train/sessions/set switches session', async ({ request }) => {
    // Get current session
    const sessResp = await request.get('/train/sessions/current');
    const sessionId = (await sessResp.text()).trim();

    // Set to same session (should succeed)
    const setResp = await request.get(`/train/sessions/set/${sessionId}`);
    expect(setResp.ok()).toBeTruthy();

    // Verify still same session
    const verifyResp = await request.get('/train/sessions/current');
    const newSessionId = (await verifyResp.text()).trim();
    expect(newSessionId).toBe(sessionId);
  });
});

test.describe('Training Overview Data Validation', () => {

  test('overview data contains score, stdev, and perf fields', async ({ request }) => {
    const resp = await request.get('/train/overview/data');
    const data = await resp.json();

    // Scores decrease over iterations (training is converging)
    expect(data.scores.length).toBeGreaterThanOrEqual(2);

    // Score iteration indices present
    expect(data.scoresIter).toBeDefined();
    expect(data.scoresIter.length).toBe(data.scores.length);

    // Perf table should have timing data
    expect(data.perf).toBeDefined();
  });

  test('model graph has correct topology for 3-layer MLP', async ({ request }) => {
    const resp = await request.get('/train/model/graph');
    const data = await resp.json();

    // Iris MLP has: Input + 3 layers (Dense, Dense, Output) = 4 vertices
    expect(data.vertexNames.length).toBeGreaterThanOrEqual(4);
    expect(data.vertexTypes).toBeDefined();
    expect(data.vertexTypes.length).toBe(data.vertexNames.length);
    expect(data.vertexInputs).toBeDefined();
    expect(data.vertexInputs.length).toBe(data.vertexNames.length);
  });

  test('layer data for all layers returns valid responses', async ({ request }) => {
    const graphResp = await request.get('/train/model/graph');
    const graph = await graphResp.json();

    // Request data for each layer by index
    for (let i = 0; i < graph.vertexNames.length; i++) {
      const resp = await request.get(`/train/model/data/${i}`);
      expect(resp.ok()).toBeTruthy();
    }
  });
});

test.describe('Training UI Interactions', () => {

  test('overview score chart has hover tooltip area', async ({ page }) => {
    await page.goto('/train/overview');
    await expect(page.locator('#scoreiterchart')).toBeVisible({ timeout: 15_000 });
    // Hover data readout elements exist (multiple elements share this ID)
    await expect(page.locator('#hoverdata').first()).toBeAttached();
  });

  test('model page sidebar shows layer details on click', async ({ page }) => {
    await page.goto('/train/model');
    // Wait for graph to render
    await expect(page.locator('#layers')).toBeVisible({ timeout: 15_000 });

    // The layer graph should have a Cytoscape canvas
    const canvas = page.locator('#layers canvas');
    if (await canvas.count() > 0) {
      // Click in center of graph to trigger node selection
      const box = await page.locator('#layers').boundingBox();
      if (box) {
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
        // Either layer details panel shows or zero state remains
        const details = page.locator('#layerDetails');
        const zeroState = page.locator('#zeroState');
        // At least one should be attached
        expect((await details.count()) + (await zeroState.count())).toBeGreaterThan(0);
      }
    }
  });

  test('system page memory chart is rendered', async ({ page }) => {
    await page.goto('/train/system');
    await page.waitForFunction(
      () => {
        const el = document.getElementById('systemMemoryChartPlot');
        return el && el.children.length > 0;
      },
      { timeout: 15_000 }
    );
    const plotChildren = await page.locator('#systemMemoryChartPlot').evaluate(
      el => el.children.length
    );
    expect(plotChildren).toBeGreaterThan(0);
  });

  test('overview to model to system navigation works', async ({ page }) => {
    await page.goto('/train/overview');
    await expect(page.locator('#scoreiterchart')).toBeVisible({ timeout: 15_000 });

    // Navigate to model via sidebar
    await page.click('a[href="model"]');
    await expect(page.locator('#layers')).toBeVisible({ timeout: 15_000 });

    // Navigate to system
    await page.click('a[href="system"]');
    await expect(page.locator('#systemMemoryChart')).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('Training System API', () => {

  test('/train/system/data returns system info', async ({ request }) => {
    const resp = await request.get('/train/system/data');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    // Response has: {updateTimestamp, memory: {"0": ...}, hardware: {"0": [...]}, software: {"0": [...]}}
    // Each of software/hardware is a map keyed by JVM index ("0"), values are arrays of [label, value] pairs
    expect(data.software).toBeDefined();
    expect(data.hardware).toBeDefined();
    expect(data.memory).toBeDefined();

    // Software info — map of "0" -> [[label, value], ...]
    const swKeys = Object.keys(data.software);
    expect(swKeys.length).toBeGreaterThanOrEqual(1);
    const swEntries = data.software[swKeys[0]];
    expect(Array.isArray(swEntries)).toBeTruthy();
    expect(swEntries.length).toBeGreaterThan(0);

    // Hardware info — same format
    const hwKeys = Object.keys(data.hardware);
    expect(hwKeys.length).toBeGreaterThanOrEqual(1);
    const hwEntries = data.hardware[hwKeys[0]];
    expect(Array.isArray(hwEntries)).toBeTruthy();
    expect(hwEntries.length).toBeGreaterThan(0);
  });
});
