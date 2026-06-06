import { test, expect, Page } from '@playwright/test';

// SameDiff page has synchronous <script> tags that reference webjars not in the test
// server's classpath (flatbuffers, webcola, weaverjs, cytoscape-spread, cytoscape-cola,
// cytoscape-euler). The Vert.x server doesn't properly close responses for missing resources,
// causing the browser to hang. We intercept and abort these requests so the page can load.
const MISSING_SCRIPTS = [
  'flatbuffers', 'webcola', 'weaverjs', 'cytoscape-spread',
  'cytoscape-cola', 'cytoscape-euler'
];

async function gotoSameDiff(page: Page) {
  // Abort requests for missing webjars so synchronous <script> tags don't hang the page
  await page.route('**/webjars/**', route => {
    const url = route.request().url();
    if (MISSING_SCRIPTS.some(s => url.includes(s))) {
      return route.abort();
    }
    return route.continue();
  });
  await page.goto('/samediff');
}

test.describe('SameDiff Page', () => {

  test('loads and shows title', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page).toHaveTitle('SameDiff Graph Visualization');
  });

  test('navbar is visible', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page.locator('.navbar')).toBeVisible({ timeout: 10_000 });
  });

  test('nav tabs are present', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page.locator('#sdnavgraph')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('#sdnavplots')).toBeVisible();
    await expect(page.locator('#sdnaveval')).toBeVisible();
    await expect(page.locator('#sdnavperf')).toBeVisible();
  });

  test('file selector is present', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page.locator('#fileselect')).toBeAttached();
  });

  test('no file loaded message is shown', async ({ page }) => {
    await gotoSameDiff(page);
    const selectedFile = page.locator('#selectedfile');
    await expect(selectedFile).toBeVisible({ timeout: 10_000 });
    const text = await selectedFile.textContent();
    expect(text).toContain('No File');
  });

  test('graph container is present', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page.locator('#graphdiv')).toBeAttached();
  });

  test('search input is present', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page.locator('#findnodetxt')).toBeVisible({ timeout: 10_000 });
  });

  test('layout radio buttons are present', async ({ page }) => {
    await gotoSameDiff(page);
    const radios = page.locator('#sidebarbottom input[type="radio"]');
    const count = await radios.count();
    expect(count).toBeGreaterThanOrEqual(3);
  });

  test('DSP nav link is present', async ({ page }) => {
    await gotoSameDiff(page);
    const dspLink = page.locator('a[href="/dsp"]');
    await expect(dspLink).toBeVisible({ timeout: 10_000 });
  });

  test('main content area is present', async ({ page }) => {
    await gotoSameDiff(page);
    await expect(page.locator('#samediffcontent')).toBeVisible({ timeout: 10_000 });
  });
});
