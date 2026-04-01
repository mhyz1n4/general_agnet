# Redis Storage Design for Short-Term Memory

## 1. Overview
The Redis storage implementation serves as the **Short-Term Memory** for the agent. It stores active conversation history and recent interactions to provide fast access for context injection.

## 2. Schema
Data will be stored as JSON-encoded strings to maintain compatibility with the `BaseStorage` interface while allowing for complex metadata.

### Item Schema (Message)
```json
{
  "id": "msg_12345",
  "role": "user | assistant | system",
  "content": "The actual message text...",
  "timestamp": 1711812000,
  "metadata": {
    "tokens": 42,
    "model": "gpt-4o",
    "intent": "recall_history"
  }
}
```

## 3. Data Structure Choices & Trade-offs

| Data Structure | Implementation | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Strings** | `SET mem:{key} '{...}'` | Simplest; direct mapping to `BaseStorage.save(key, data)`. | High overhead for "get all session messages" (requires `KEYS` or `SCAN`). |
| **Hashes** | `HSET agent:memory {key} '{...}'` | Groups all memory items in one key; memory efficient for small items. | No built-in ordering for range-based retrieval. |
| **Lists** | `RPUSH session:{id} '{...}'` | Maintains strict chronological order; O(1) appends. | No random access by `message_id`; hard to update specific messages. |
| **Sorted Sets** | `ZADD session:{id} {ts} '{...}'` | Automatically ordered by timestamp; supports range queries (e.g., "last 10 mins"). | Slightly higher memory overhead; complex to update items. |

## 4. Selection: Redis Hash with Key Prefixes
For the v1 implementation of `BaseStorage`, we will use **Redis Strings with a configurable prefix** (defaulting to `mem:`). 

### Rationale
1. **Interface Compatibility**: `BaseStorage` is a simple Key-Value interface. Redis Strings map perfectly to `save(key, data)` and `load(key)`.
2. **TTL Support**: Redis Strings allow setting an Expiration (TTL) per key. This is critical for "Short-Term Memory" to ensure old data is automatically purged, keeping the "active window" fresh.
3. **Atomicity**: `SET` and `GET` are atomic and fast.

## 5. Session Management (Future Consideration)
While `BaseStorage` handles individual items, a `SessionManager` could use a **Redis List** to store a sequence of `message_id`s for a specific session, enabling fast retrieval of the "Active Window" while keeping the actual content in the main KV store.
