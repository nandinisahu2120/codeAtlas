package cache

import (
	"sync"
	"time"
)

type CacheItem struct {
	Value     string
	ExpiresAt time.Time
}

type MemoryCache struct {
	mu    sync.RWMutex
	items map[string]CacheItem
}

func NewMemoryCache() *MemoryCache {
	return &MemoryCache{
		items: make(map[string]CacheItem),
	}
}

func (c *MemoryCache) Set(key string, value string, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[key] = CacheItem{
		Value:     value,
		ExpiresAt: time.Now().Add(ttl),
	}
}

func (c *MemoryCache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	item, found := c.items[key]
	if !found {
		return "", false
	}
	if time.Now().After(item.ExpiresAt) {
		return "", false
	}
	return item.Value, true
}
