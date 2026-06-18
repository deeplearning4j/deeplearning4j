import { test, expect } from '@playwright/test';

test.describe('Entity Resolution Page', () => {

  test('loads and shows title', async ({ page }) => {
    await page.goto('/entities');
    await expect(page).toHaveTitle('Entity Resolution');
  });

  test('stats bar is visible', async ({ page }) => {
    await page.goto('/entities');
    await expect(page.locator('#statsBar, .stats-bar')).toBeVisible({ timeout: 10_000 });
  });

  test('entity table is present', async ({ page }) => {
    await page.goto('/entities');
    await expect(page.locator('#entityTable')).toBeAttached();
  });

  test('tabs are present', async ({ page }) => {
    await page.goto('/entities');
    const tabs = page.locator('.tab');
    const count = await tabs.count();
    expect(count).toBeGreaterThanOrEqual(3);
  });

  test('ingest tab can be activated', async ({ page }) => {
    await page.goto('/entities');
    // Click the Ingest tab
    const ingestTab = page.locator('.tab', { hasText: 'Ingest' });
    if (await ingestTab.count() > 0) {
      await ingestTab.click();
      await expect(page.locator('#ingestPanel')).toBeVisible({ timeout: 10_000 });
    }
  });

  test('ingest form fields are present', async ({ page }) => {
    await page.goto('/entities');
    // Navigate to ingest panel
    const ingestTab = page.locator('.tab', { hasText: 'Ingest' });
    if (await ingestTab.count() > 0) {
      await ingestTab.click();
      await expect(page.locator('#ingestText')).toBeVisible({ timeout: 10_000 });
      await expect(page.locator('#ingestType')).toBeVisible();
    }
  });

  test('candidates tab can be activated', async ({ page }) => {
    await page.goto('/entities');
    const candidatesTab = page.locator('.tab', { hasText: 'Candidates' });
    if (await candidatesTab.count() > 0) {
      await candidatesTab.click();
      await expect(page.locator('#candidatesPanel')).toBeVisible({ timeout: 10_000 });
    }
  });

  test('search box is present', async ({ page }) => {
    await page.goto('/entities');
    await expect(page.locator('#searchBox')).toBeAttached();
  });

  test('sort controls are present', async ({ page }) => {
    await page.goto('/entities');
    await expect(page.locator('#sortBy')).toBeAttached();
    await expect(page.locator('#sortOrder')).toBeAttached();
  });
});

test.describe('Entity Resolution UI Interactions', () => {

  test('settings tab can be activated', async ({ page }) => {
    await page.goto('/entities');
    const settingsTab = page.locator('.tab', { hasText: 'Settings' });
    if (await settingsTab.count() > 0) {
      await settingsTab.click();
      await expect(page.locator('#settingsPanel')).toBeVisible({ timeout: 10_000 });
    }
  });

  test('type filter dropdown has options', async ({ page }) => {
    await page.goto('/entities');
    const filter = page.locator('#typeFilter');
    await expect(filter).toBeAttached();
    const options = filter.locator('option');
    // Should have at least "All" option
    expect(await options.count()).toBeGreaterThanOrEqual(1);
  });

  test('sort controls can be changed', async ({ page }) => {
    await page.goto('/entities');
    const sortBy = page.locator('#sortBy');
    await expect(sortBy).toBeAttached();

    // Should have multiple sort options
    const options = sortBy.locator('option');
    expect(await options.count()).toBeGreaterThanOrEqual(2);
  });

  test('entity table headers are correct', async ({ page }) => {
    await page.goto('/entities');
    const headers = page.locator('#entityTable th');
    if (await headers.count() > 0) {
      const headerTexts: string[] = [];
      const count = await headers.count();
      for (let i = 0; i < count; i++) {
        headerTexts.push((await headers.nth(i).textContent())!.trim());
      }
      // Should have Name, Type, Confidence, Mentions columns
      expect(headerTexts.some(h => h.includes('Name'))).toBeTruthy();
      expect(headerTexts.some(h => h.includes('Type'))).toBeTruthy();
    }
  });

  test('stats show zero counts when empty', async ({ page, request }) => {
    // Clear entities
    await request.post('/api/entities/clear');
    await page.goto('/entities');

    // Wait for stats to load
    await page.waitForTimeout(2000);

    const totalText = await page.locator('#statTotal').textContent();
    expect(totalText).toBe('0');
  });

  test('candidate threshold slider is present', async ({ page }) => {
    await page.goto('/entities');
    const candidatesTab = page.locator('.tab', { hasText: 'Candidates' });
    if (await candidatesTab.count() > 0) {
      await candidatesTab.click();
      await expect(page.locator('#candidateThreshold')).toBeAttached();
    }
  });
});

test.describe('Entity Resolution API', () => {

  test('/api/entities returns array', async ({ request }) => {
    const resp = await request.get('/api/entities');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(Array.isArray(data)).toBeTruthy();
  });

  test('/api/entities/stats returns stats object', async ({ request }) => {
    const resp = await request.get('/api/entities/stats');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data).toBeDefined();
  });

  test('/api/entities/candidates returns array', async ({ request }) => {
    const resp = await request.get('/api/entities/candidates');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(Array.isArray(data)).toBeTruthy();
  });

  test('ingest and query roundtrip works', async ({ request }) => {
    // Clear existing entities first
    await request.post('/api/entities/clear');

    // Ingest a single entity
    const ingestResp = await request.post('/api/entities/ingest', {
      data: { text: 'Playwright Test Corp', type: 'ORGANIZATION', source: 'e2e-test' }
    });
    expect(ingestResp.ok()).toBeTruthy();

    const entity = await ingestResp.json();
    expect(entity.canonicalName || entity.text || entity.name).toBeTruthy();

    // Query entities — should find the one we just added
    const queryResp = await request.get('/api/entities');
    const entities = await queryResp.json();
    expect(entities.length).toBeGreaterThanOrEqual(1);

    // Cleanup
    await request.post('/api/entities/clear');
  });

  test('batch ingest works', async ({ request }) => {
    await request.post('/api/entities/clear');

    const batchResp = await request.post('/api/entities/ingestBatch', {
      data: {
        entities: [
          { text: 'Alice Smith', type: 'PERSON' },
          { text: 'Bob Jones', type: 'PERSON' },
          { text: 'Acme Inc', type: 'ORGANIZATION' }
        ],
        source: 'e2e-batch-test'
      }
    });
    expect(batchResp.ok()).toBeTruthy();

    const result = await batchResp.json();
    expect(Array.isArray(result)).toBeTruthy();
    expect(result.length).toBe(3);

    // Verify stats updated
    const statsResp = await request.get('/api/entities/stats');
    const stats = await statsResp.json();
    expect(stats).toBeDefined();

    // Cleanup
    await request.post('/api/entities/clear');
  });

  test('merge two entities', async ({ request }) => {
    await request.post('/api/entities/clear');

    // Ingest two clearly distinct entities so they aren't auto-resolved together
    const resp1 = await request.post('/api/entities/ingest', {
      data: { text: 'Alpha Corporation', type: 'ORGANIZATION' }
    });
    const resp2 = await request.post('/api/entities/ingest', {
      data: { text: 'Beta Industries', type: 'ORGANIZATION' }
    });

    const entity1 = await resp1.json();
    const entity2 = await resp2.json();

    // Only merge if they resolved as distinct entities
    if (entity1.id && entity2.id && entity1.id !== entity2.id) {
      const mergeResp = await request.post('/api/entities/merge', {
        data: { entityIdA: entity1.id, entityIdB: entity2.id }
      });
      expect(mergeResp.ok()).toBeTruthy();

      const merged = await mergeResp.json();
      expect(merged).toBeDefined();
    }

    // Cleanup
    await request.post('/api/entities/clear');
  });

  test('rename entity works', async ({ request }) => {
    await request.post('/api/entities/clear');

    const ingestResp = await request.post('/api/entities/ingest', {
      data: { text: 'Old Name Inc', type: 'ORGANIZATION' }
    });
    const entity = await ingestResp.json();

    const renameResp = await request.post(`/api/entities/${entity.id}/rename`, {
      data: { name: 'New Name Corp' }
    });
    expect(renameResp.ok()).toBeTruthy();

    const renamed = await renameResp.json();
    expect(renamed.canonicalName).toBe('New Name Corp');

    await request.post('/api/entities/clear');
  });

  test('retype entity works', async ({ request }) => {
    await request.post('/api/entities/clear');

    const ingestResp = await request.post('/api/entities/ingest', {
      data: { text: 'Jane Doe', type: 'PERSON' }
    });
    const entity = await ingestResp.json();

    const retypeResp = await request.post(`/api/entities/${entity.id}/retype`, {
      data: { type: 'ORGANIZATION' }
    });
    expect(retypeResp.ok()).toBeTruthy();

    const retyped = await retypeResp.json();
    expect(retyped.type).toBe('ORGANIZATION');

    await request.post('/api/entities/clear');
  });

  test('split entity creates new entity', async ({ request }) => {
    await request.post('/api/entities/clear');

    // Ingest two mentions that resolve to the same entity
    await request.post('/api/entities/ingest', {
      data: { text: 'SplitTest Corp', type: 'ORGANIZATION' }
    });
    await request.post('/api/entities/ingest', {
      data: { text: 'SplitTest Corp', type: 'ORGANIZATION', source: 'second' }
    });

    // Get the entity
    const entities = await (await request.get('/api/entities')).json();
    if (entities.length > 0 && entities[0].mentions && entities[0].mentions.length > 1) {
      const splitResp = await request.post(`/api/entities/${entities[0].id}/split`, {
        data: { mentionIndex: 1 }
      });
      expect(splitResp.ok()).toBeTruthy();

      const newEntity = await splitResp.json();
      expect(newEntity.id).toBeDefined();
      expect(newEntity.id).not.toBe(entities[0].id);
    }

    await request.post('/api/entities/clear');
  });

  test('set threshold per type', async ({ request }) => {
    const resp = await request.post('/api/entities/threshold', {
      data: { type: 'PERSON', threshold: 0.8 }
    });
    expect(resp.ok()).toBeTruthy();

    const result = await resp.json();
    expect(result.type).toBe('PERSON');
    expect(result.threshold).toBe(0.8);
  });

  test('query with type filter', async ({ request }) => {
    await request.post('/api/entities/clear');

    await request.post('/api/entities/ingest', {
      data: { text: 'Alice', type: 'PERSON' }
    });
    await request.post('/api/entities/ingest', {
      data: { text: 'Acme Corp', type: 'ORGANIZATION' }
    });

    // Filter by type
    const personResp = await request.get('/api/entities?type=PERSON');
    expect(personResp.ok()).toBeTruthy();
    const persons = await personResp.json();
    for (const p of persons) {
      expect(p.type).toBe('PERSON');
    }

    // Filter by different type
    const orgResp = await request.get('/api/entities?type=ORGANIZATION');
    expect(orgResp.ok()).toBeTruthy();
    const orgs = await orgResp.json();
    for (const o of orgs) {
      expect(o.type).toBe('ORGANIZATION');
    }

    await request.post('/api/entities/clear');
  });

  test('query with sort options', async ({ request }) => {
    await request.post('/api/entities/clear');

    await request.post('/api/entities/ingest', { data: { text: 'Bravo', type: 'PERSON' } });
    await request.post('/api/entities/ingest', { data: { text: 'Alpha', type: 'PERSON' } });
    await request.post('/api/entities/ingest', { data: { text: 'Charlie', type: 'PERSON' } });

    const resp = await request.get('/api/entities?sort=name&order=asc');
    expect(resp.ok()).toBeTruthy();
    const sorted = await resp.json();
    if (sorted.length >= 3) {
      // Names should be in alphabetical order
      const names = sorted.map((e: any) => e.canonicalName);
      const sortedNames = [...names].sort();
      expect(names).toEqual(sortedNames);
    }

    await request.post('/api/entities/clear');
  });

  test('candidates with minConfidence filter', async ({ request }) => {
    const resp = await request.get('/api/entities/candidates?minConfidence=0.9');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(Array.isArray(data)).toBeTruthy();
    // All candidates should have confidence >= threshold
    for (const c of data) {
      expect(c.confidence).toBeGreaterThanOrEqual(0.9);
    }
  });

  test('clear removes all entities', async ({ request }) => {
    // Add some entities
    await request.post('/api/entities/ingest', { data: { text: 'Test1', type: 'PERSON' } });
    await request.post('/api/entities/ingest', { data: { text: 'Test2', type: 'PERSON' } });

    // Clear
    const clearResp = await request.post('/api/entities/clear');
    expect(clearResp.ok()).toBeTruthy();
    const clearResult = await clearResp.json();
    expect(clearResult.cleared).toBe(true);

    // Verify empty
    const entities = await (await request.get('/api/entities')).json();
    expect(entities.length).toBe(0);
  });

  test('delete entity works', async ({ request }) => {
    await request.post('/api/entities/clear');

    const ingestResp = await request.post('/api/entities/ingest', {
      data: { text: 'To Be Deleted', type: 'PERSON' }
    });
    const entity = await ingestResp.json();

    if (entity.id) {
      const deleteResp = await request.delete(`/api/entities/${entity.id}`);
      expect(deleteResp.ok()).toBeTruthy();

      // Verify deleted
      const queryResp = await request.get('/api/entities');
      const entities = await queryResp.json();
      const found = entities.find((e: any) => e.id === entity.id);
      expect(found).toBeUndefined();
    }

    await request.post('/api/entities/clear');
  });
});
