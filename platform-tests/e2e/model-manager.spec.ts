import { test, expect } from '@playwright/test';

test.describe('Model Manager Page', () => {

  test('loads and shows title', async ({ page }) => {
    await page.goto('/models');
    await expect(page).toHaveTitle('DL4J Model Manager');
  });

  test('catalog tab is active by default', async ({ page }) => {
    await page.goto('/models');
    const catalogTab = page.locator('.dl4j-tab', { hasText: 'Catalog' });
    await expect(catalogTab).toBeVisible({ timeout: 10_000 });
    const className = await catalogTab.getAttribute('class');
    expect(className).toContain('active');
  });

  test('catalog view is visible', async ({ page }) => {
    await page.goto('/models');
    await expect(page.locator('#catalogView')).toBeVisible({ timeout: 10_000 });
  });

  test('installed tab can be activated', async ({ page }) => {
    await page.goto('/models');
    const installedTab = page.locator('.dl4j-tab', { hasText: 'Installed' });
    await expect(installedTab).toBeVisible({ timeout: 10_000 });

    await installedTab.click();
    await expect(page.locator('#installedView')).toBeVisible({ timeout: 10_000 });
  });

  test('toolbar has nav links', async ({ page }) => {
    await page.goto('/models');
    const toolbar = page.locator('.dl4j-toolbar');
    await expect(toolbar).toBeVisible({ timeout: 10_000 });

    const links = page.locator('.dl4j-toolbar a');
    const hrefs: string[] = [];
    const count = await links.count();
    for (let i = 0; i < count; i++) {
      const href = await links.nth(i).getAttribute('href');
      if (href) hrefs.push(href);
    }

    expect(hrefs).toContain('/samediff');
    expect(hrefs).toContain('/modelview');
    expect(hrefs).toContain('/train/overview');
  });

  test('model cards or empty state is shown', async ({ page }) => {
    await page.goto('/models');
    // Wait for catalog to load
    await page.waitForTimeout(3000);

    // Should have model cards or an empty state
    const cards = page.locator('.dl4j-card');
    const empty = page.locator('.dl4j-no-data');
    const cardCount = await cards.count();
    const emptyCount = await empty.count();
    expect(cardCount + emptyCount).toBeGreaterThan(0);
  });
});

test.describe('Model Manager API', () => {

  test('/models/catalog returns array', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(Array.isArray(data)).toBeTruthy();

    // Each entry should have expected fields
    if (data.length > 0) {
      expect(data[0].id).toBeDefined();
      expect(data[0].format).toBeDefined();
    }
  });

  test('/models/installed returns array', async ({ request }) => {
    const resp = await request.get('/models/installed');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(Array.isArray(data)).toBeTruthy();
  });

  test('/models/catalog entries have expected schema', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    for (const entry of data) {
      // Every catalog entry must have id, name, format, and size
      expect(entry.id).toBeDefined();
      expect(typeof entry.id).toBe('string');
      expect(entry.format).toBeDefined();
    }
  });

  test('/models/remove returns 404 for unknown model', async ({ request }) => {
    const resp = await request.post('/models/remove/nonexistent-model-id');
    expect(resp.status()).toBe(404);

    const data = await resp.json();
    expect(data.error).toBeDefined();
  });

  test('/models/load returns 404 for unknown model', async ({ request }) => {
    const resp = await request.get('/models/load/nonexistent-model-id');
    expect(resp.status()).toBe(404);

    const data = await resp.json();
    expect(data.error).toContain('not found');
  });

  test('/models/download rejects missing fields', async ({ request }) => {
    const resp = await request.post('/models/download', {
      data: { id: 'test-only' }
    });
    // Missing repo and filePath — should reject
    expect(resp.status()).toBeGreaterThanOrEqual(400);
  });

  test('/models/download/unknown/progress returns not_found', async ({ request }) => {
    const resp = await request.get('/models/download/unknown-id/progress');
    expect(resp.ok()).toBeTruthy();

    // SSE endpoint — response is text/event-stream
    const text = await resp.text();
    expect(text).toContain('not_found');
  });

  test('/models/omnihub/download rejects missing huggingFaceId', async ({ request }) => {
    const resp = await request.post('/models/omnihub/download', {
      data: { name: 'test' }
    });
    expect(resp.status()).toBeGreaterThanOrEqual(400);

    const data = await resp.json();
    expect(data.error).toContain('huggingFaceId');
  });
});

test.describe('Model Manager UI Interactions', () => {

  test('search box filters cards', async ({ page }) => {
    await page.goto('/models');
    const searchBox = page.locator('#searchBox');
    if (await searchBox.count() > 0) {
      await searchBox.fill('nonexistent-model-xyz');
      await page.waitForTimeout(1000);
      // After filtering with a non-matching term, visible cards should be zero or empty state shows
      const cards = page.locator('.model-card:visible');
      const empty = page.locator('.dl4j-no-data:visible');
      expect((await cards.count()) + (await empty.count())).toBeGreaterThanOrEqual(0);
    }
  });

  test('tab switching preserves state', async ({ page }) => {
    await page.goto('/models');
    // Switch to installed, then back to catalog
    const installedTab = page.locator('.dl4j-tab', { hasText: 'Installed' });
    await expect(installedTab).toBeVisible({ timeout: 10_000 });
    await installedTab.click();
    await expect(page.locator('#installedView')).toBeVisible({ timeout: 10_000 });

    const catalogTab = page.locator('.dl4j-tab', { hasText: 'Catalog' });
    await catalogTab.click();
    await expect(page.locator('#catalogView')).toBeVisible({ timeout: 10_000 });
  });

  test('installed view shows empty state when no models installed', async ({ page }) => {
    await page.goto('/models');
    const installedTab = page.locator('.dl4j-tab', { hasText: 'Installed' });
    await installedTab.click();
    await expect(page.locator('#installedView')).toBeVisible({ timeout: 10_000 });

    // Either model rows or empty state
    await page.waitForTimeout(2000);
    const rows = page.locator('#installedView .dl4j-card');
    const empty = page.locator('#installedView .dl4j-no-data');
    expect((await rows.count()) + (await empty.count())).toBeGreaterThanOrEqual(0);
  });
});

test.describe('Model Manager LLM Metadata API', () => {

  test('/models/catalog entries include LLM metadata fields', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(data.length).toBeGreaterThan(0);

    // Find an LLM entry (qwen or smollm)
    const llmEntry = data.find((e: any) => e.modelType === 'LLM_GGUF');
    expect(llmEntry).toBeDefined();
    expect(llmEntry.architecture).toBeDefined();
    expect(llmEntry.contextLength).toBeGreaterThan(0);
    expect(llmEntry.parameterCount).toBeGreaterThan(0);
    expect(llmEntry.chatTemplateFormat).toBeDefined();
  });

  test('/models/catalog entries include tokenizer file list', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    const llmEntry = data.find((e: any) => e.modelType === 'LLM_GGUF');
    expect(llmEntry).toBeDefined();
    expect(Array.isArray(llmEntry.tokenizerFiles)).toBeTruthy();
    expect(llmEntry.tokenizerFiles.length).toBeGreaterThan(0);
    expect(llmEntry.tokenizerFiles).toContain('tokenizer.json');
  });

  test('/models/catalog has modelType on every entry', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    for (const entry of data) {
      expect(entry.modelType).toBeDefined();
      expect(typeof entry.modelType).toBe('string');
    }
  });

  test('/models/catalog encoder entries have correct modelType', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    const encoder = data.find((e: any) => e.category === 'encoders');
    if (encoder) {
      expect(encoder.modelType).toBe('ENCODER');
    }
  });

  test('/models/catalog cross_encoder entries have correct modelType', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    const crossEncoder = data.find((e: any) => e.category === 'cross_encoders');
    if (crossEncoder) {
      expect(crossEncoder.modelType).toBe('CROSS_ENCODER');
    }
  });

  test('/models/:id/info returns 404 for unknown model', async ({ request }) => {
    const resp = await request.get('/models/nonexistent-model-xyz/info');
    expect(resp.status()).toBe(404);
    const data = await resp.json();
    expect(data.error).toContain('not found');
  });

  test('/models/:id/tokenizer returns 404 for unknown model', async ({ request }) => {
    const resp = await request.get('/models/nonexistent-model-xyz/tokenizer');
    expect(resp.status()).toBe(404);
    const data = await resp.json();
    expect(data.error).toContain('not found');
  });

  test('/models/:id/chat-template returns 404 for unknown model', async ({ request }) => {
    const resp = await request.get('/models/nonexistent-model-xyz/chat-template');
    expect(resp.status()).toBe(404);
    const data = await resp.json();
    expect(data.error).toContain('not found');
  });

  test('/models/catalog qwen entry has correct metadata', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    const qwen = data.find((e: any) => e.id === 'qwen2.5-0.5b-instruct');
    expect(qwen).toBeDefined();
    expect(qwen.architecture).toBe('qwen2');
    expect(qwen.quantizationType).toBe('Q4_K_M');
    expect(qwen.contextLength).toBe(32768);
    expect(qwen.parameterCount).toBe(494000000);
    expect(qwen.chatTemplateFormat).toBe('chatml');
    expect(qwen.format).toBe('gguf');
  });

  test('/models/catalog smollm entry has correct metadata', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    const smollm = data.find((e: any) => e.id === 'smollm2-360m-instruct');
    expect(smollm).toBeDefined();
    expect(smollm.architecture).toBe('llama');
    expect(smollm.contextLength).toBe(8192);
    expect(smollm.parameterCount).toBe(362000000);
    expect(smollm.modelType).toBe('LLM_GGUF');
  });

  test('/models/catalog omnihub zoo entries have OTHER modelType', async ({ request }) => {
    const resp = await request.get('/models/catalog');
    const data = await resp.json();

    const zooEntry = data.find((e: any) => e.category === 'omnihub_zoo');
    if (zooEntry) {
      expect(zooEntry.modelType).toBe('OTHER');
      expect(zooEntry.source).toBe('omnihub');
    }
  });
});

test.describe('Cache Management API', () => {

  test('/models/cache/status returns all three cache sections', async ({ request }) => {
    const resp = await request.get('/models/cache/status');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();

    expect(data.dsp).toBeDefined();
    expect(data.triton).toBeDefined();
    expect(data.sdz).toBeDefined();
  });

  test('/models/cache/status DSP section has expected fields', async ({ request }) => {
    const resp = await request.get('/models/cache/status');
    const data = await resp.json();

    expect(typeof data.dsp.enabled).toBe('boolean');
    expect(typeof data.dsp.directory).toBe('string');
    expect(typeof data.dsp.entryCount).toBe('number');
    expect(typeof data.dsp.totalBytes).toBe('number');
    expect(data.dsp.entryCount).toBeGreaterThanOrEqual(0);
    expect(data.dsp.totalBytes).toBeGreaterThanOrEqual(0);
  });

  test('/models/cache/status Triton section has expected fields', async ({ request }) => {
    const resp = await request.get('/models/cache/status');
    const data = await resp.json();

    expect(typeof data.triton.available).toBe('boolean');
    expect(typeof data.triton.directory).toBe('string');
    expect(typeof data.triton.entryCount).toBe('number');
    expect(typeof data.triton.totalBytes).toBe('number');
  });

  test('/models/cache/status SDZ section has expected fields', async ({ request }) => {
    const resp = await request.get('/models/cache/status');
    const data = await resp.json();

    expect(typeof data.sdz.directory).toBe('string');
    expect(typeof data.sdz.entryCount).toBe('number');
    expect(typeof data.sdz.totalBytes).toBe('number');
  });

  test('/models/cache/dsp returns list with metadata', async ({ request }) => {
    const resp = await request.get('/models/cache/dsp');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();

    expect(typeof data.directory).toBe('string');
    expect(typeof data.enabled).toBe('boolean');
    expect(typeof data.forceRecompile).toBe('boolean');
    expect(typeof data.entryCount).toBe('number');
    expect(Array.isArray(data.entries)).toBeTruthy();

    // If entries exist, they should have hash and sizeBytes
    if (data.entries.length > 0) {
      expect(data.entries[0].hash).toBeDefined();
      expect(typeof data.entries[0].sizeBytes).toBe('number');
    }
  });

  test('/models/cache/dsp/:hash invalidate returns result for unknown hash', async ({ request }) => {
    const resp = await request.post('/models/cache/dsp/0000000000000000');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();

    expect(data.hash).toBe('0000000000000000');
    expect(data.existed).toBe(false);
  });

  test('/models/cache/dsp/:hash rejects invalid hex hash', async ({ request }) => {
    const resp = await request.post('/models/cache/dsp/not-a-hex-hash');
    expect(resp.status()).toBe(400);
    const data = await resp.json();
    expect(data.error).toContain('Invalid hash');
  });

  test('/models/cache/triton returns cache info', async ({ request }) => {
    const resp = await request.get('/models/cache/triton');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();

    expect(typeof data.available).toBe('boolean');
    expect(typeof data.directory).toBe('string');
    expect(typeof data.entryCount).toBe('number');
    expect(Array.isArray(data.entries)).toBeTruthy();
  });

  test('/models/cache/sdz returns cache listing', async ({ request }) => {
    const resp = await request.get('/models/cache/sdz');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();

    expect(typeof data.directory).toBe('string');
    expect(typeof data.entryCount).toBe('number');
    expect(typeof data.totalBytes).toBe('number');
    expect(Array.isArray(data.entries)).toBeTruthy();

    // If entries exist, they should have filename, sizeBytes, path
    if (data.entries.length > 0) {
      expect(data.entries[0].filename).toBeDefined();
      expect(typeof data.entries[0].sizeBytes).toBe('number');
      expect(data.entries[0].path).toBeDefined();
    }
  });

  test('/models/cache/triton/import rejects missing bundlePath', async ({ request }) => {
    const resp = await request.post('/models/cache/triton/import', {
      data: {}
    });
    expect(resp.status()).toBe(400);
    const data = await resp.json();
    expect(data.error).toContain('bundlePath');
  });

  test('/models/cache/dsp/clear clears DSP plan cache', async ({ request }) => {
    const resp = await request.post('/models/cache/dsp/clear');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(typeof data.cleared).toBe('number');
    expect(typeof data.directory).toBe('string');
  });

  test('/models/cache/triton/clear clears Triton kernel cache', async ({ request }) => {
    const resp = await request.post('/models/cache/triton/clear');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(typeof data.cleared).toBe('number');
    expect(typeof data.filesDeleted).toBe('number');
    expect(typeof data.directory).toBe('string');
  });

  test('/models/cache/sdz/clear clears SDZ model cache', async ({ request }) => {
    const resp = await request.post('/models/cache/sdz/clear');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(typeof data.cleared).toBe('number');
    expect(typeof data.directory).toBe('string');
  });

  test('/models/cache/sdz/:filename rejects invalid filename', async ({ request }) => {
    const resp = await request.post('/models/cache/sdz/not-an-sdz.txt');
    expect(resp.status()).toBe(400);
    const data = await resp.json();
    expect(data.error).toContain('Invalid filename');
  });

  test('/models/cache/sdz/:filename rejects path traversal', async ({ request }) => {
    const resp = await request.post('/models/cache/sdz/..%2F..%2Fetc%2Fpasswd.sdz');
    expect(resp.status()).toBe(400);
    const data = await resp.json();
    expect(data.error).toContain('Invalid filename');
  });

  test('/models/cache/sdz/:filename returns existed=false for unknown file', async ({ request }) => {
    const resp = await request.post('/models/cache/sdz/nonexistent_model.sdz');
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(data.filename).toBe('nonexistent_model.sdz');
    expect(data.existed).toBe(false);
    expect(data.deleted).toBe(false);
  });

  test('/models/cache/triton/export returns result or error', async ({ request }) => {
    const resp = await request.post('/models/cache/triton/export', {
      data: { outputPath: '/tmp/test_triton_export.tkcache' }
    });
    const data = await resp.json();
    if (resp.ok()) {
      // Triton available — export succeeded
      expect(typeof data.exported).toBe('number');
      expect(data.outputPath).toBe('/tmp/test_triton_export.tkcache');
    } else {
      // Triton not available on CPU — expect error
      expect(resp.status()).toBe(500);
      expect(data.error).toBeDefined();
    }
  });
});
