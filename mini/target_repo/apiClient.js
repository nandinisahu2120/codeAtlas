/**
 * API client module for fetching remote resources.
 */
export class ApiClient {
  constructor(baseUrl, timeoutMs = 5000) {
    this.baseUrl = baseUrl;
    this.timeout = timeoutMs;
  }

  async fetchEndpoint(path, options = {}) {
    const url = `${this.baseUrl}/${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  }

  ping() {
    return this.fetchEndpoint('health');
  }
}
