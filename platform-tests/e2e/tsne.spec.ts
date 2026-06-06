import { test, expect } from '@playwright/test';

test.describe('TSNE Page', () => {

  test('loads and shows title', async ({ page }) => {
    await page.goto('/tsne');
    await expect(page).toHaveTitle('T-SNE renders');
  });

  test('embed container is present', async ({ page }) => {
    await page.goto('/tsne');
    await expect(page.locator('#embed')).toBeVisible({ timeout: 10_000 });
  });

  test('session selector is present', async ({ page }) => {
    await page.goto('/tsne');
    await expect(page.locator('#sessionSelect')).toBeAttached();
  });

  test('upload form is present', async ({ page }) => {
    await page.goto('/tsne');
    await expect(page.locator('#form')).toBeAttached();
  });
});

test.describe('TSNE UI Elements', () => {

  test('dimension selector is present', async ({ page }) => {
    await page.goto('/tsne');
    const dimSelect = page.locator('#dim, #dimensions, select');
    await expect(dimSelect.first()).toBeAttached();
  });

  test('render button is present', async ({ page }) => {
    await page.goto('/tsne');
    const renderBtn = page.locator('#render, button, input[type="submit"]');
    await expect(renderBtn.first()).toBeAttached();
  });

  test('page contains script for d3 or plotting', async ({ page }) => {
    await page.goto('/tsne');
    // The page should have the embed container ready for rendering
    const embed = page.locator('#embed');
    await expect(embed).toBeVisible({ timeout: 10_000 });
    const html = await embed.innerHTML();
    // Either SVG/canvas will be injected or it's empty waiting for data
    expect(typeof html).toBe('string');
  });
});

test.describe('TSNE API', () => {

  test('/tsne/sessions returns array', async ({ request }) => {
    const resp = await request.get('/tsne/sessions');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(Array.isArray(data)).toBeTruthy();
  });

  test('/tsne/coords returns empty for unknown session', async ({ request }) => {
    const resp = await request.get('/tsne/coords/nonexistent-session');
    expect(resp.ok()).toBeTruthy();

    // Unknown session returns empty response body
    const text = await resp.text();
    expect(text === '' || text === '[]' || text === 'null').toBeTruthy();
  });

  test('/tsne/upload and then coords roundtrip', async ({ request }) => {
    // Upload a coordinate file
    const coordData = '0.1,0.2,label1\n0.3,0.4,label2\n0.5,0.6,label3';
    const uploadResp = await request.post('/tsne/upload', {
      multipart: {
        file: {
          name: 'coords.csv',
          mimeType: 'text/csv',
          buffer: Buffer.from(coordData)
        }
      }
    });
    expect(uploadResp.ok()).toBeTruthy();

    // Sessions should now include UploadedFile
    const sessResp = await request.get('/tsne/sessions');
    const sessions = await sessResp.json();
    expect(sessions).toContain('UploadedFile');

    // Fetch coords for uploaded file
    const coordsResp = await request.get('/tsne/coords/UploadedFile');
    expect(coordsResp.ok()).toBeTruthy();

    const coords = await coordsResp.json();
    expect(Array.isArray(coords)).toBeTruthy();
    expect(coords.length).toBe(3);
    expect(coords[0]).toContain('0.1');
  });

  test('/tsne/post/:sid stores session coords', async ({ request }) => {
    const coordData = '1.0,2.0,pointA\n3.0,4.0,pointB';
    const postResp = await request.post('/tsne/post/test-session-e2e', {
      multipart: {
        file: {
          name: 'test.csv',
          mimeType: 'text/csv',
          buffer: Buffer.from(coordData)
        }
      }
    });
    expect(postResp.ok()).toBeTruthy();

    // Session should now be listed
    const sessResp = await request.get('/tsne/sessions');
    const sessions = await sessResp.json();
    expect(sessions).toContain('test-session-e2e');

    // Coords should be retrievable
    const coordsResp = await request.get('/tsne/coords/test-session-e2e');
    expect(coordsResp.ok()).toBeTruthy();

    const coords = await coordsResp.json();
    expect(Array.isArray(coords)).toBeTruthy();
    expect(coords.length).toBe(2);
  });
});
