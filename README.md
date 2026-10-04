# BAYMAX

### Local-First, Privacy-Preserving, Cross-Device AI Agent

> **BAYMAX** is a local-first personal AI agent designed to work across Android and Windows devices, maintain persistent memory, synchronize information over a trusted peer-to-peer network, and execute authorized tasks with low latency — without requiring a cloud backend for its core functionality.

---

## 🚀 Vision

Most modern AI assistants depend heavily on centralized cloud infrastructure.

This creates several problems:

- Data must leave the user's devices.
- Internet connectivity becomes a dependency.
- Latency increases because requests travel to remote servers.
- Persistent personal context is fragmented across applications.
- Different devices do not share a unified local context.
- Users have limited control over where their data and memories are stored.
- Agentic systems can become difficult to control once they gain access to multiple tools.

**BAYMAX aims to approach the problem differently.**

Instead of:

```text
Device → Internet → Cloud AI → Database → Device
```

BAYMAX is designed around:

```text
              ┌───────────────┐
              │    Android    │
              │   BAYMAX App  │
              └───────┬───────┘
                      │
                Secure P2P
                      │
                      ▼
              ┌───────────────┐
              │    Windows    │
              │  BAYMAX Core  │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    AI Engine      Memory          Tasks
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              Local User Context
```

The initial architecture is **P2P-first** and can later evolve into a **hybrid architecture** for remote connectivity.

---

# 🎯 Core Objectives

BAYMAX aims to provide:

- 🔒 Local-first data processing
- 🧠 Persistent personal memory
- 📱 Android + Windows support
- 🔄 Cross-device synchronization
- ⚡ Low-latency local AI
- 🤖 Agentic task execution
- 🛡️ Capability-based security
- 🌐 Peer-to-peer communication
- 📦 Offline-first operation
- 🔍 RAG-based contextual retrieval
- 🧩 Modular architecture
- 🧪 Fault-tolerant execution
- 🔌 Replaceable AI/network/memory components

---

# ✨ Key Features

## 1. Cross-Device Context

BAYMAX allows different devices to contribute information to a shared personal context.

For example:

```text
Android
   │
   ├── User interaction
   ├── Selected files
   ├── Device context
   └── User-approved information
            │
            ▼
        Local Store
            │
            ▼
        P2P Sync
            │
            ▼
        Windows Core
```

The system does **not** assume that every device is always online.

---

# 2. Local-First Persistent Memory

BAYMAX maintains persistent memory locally.

Memory can be divided into:

### Working Memory

Short-lived context required for the current task.

### Episodic Memory

Records of previous interactions and completed tasks.

### Semantic Memory

Long-term information that BAYMAX has been explicitly allowed to retain.

Every durable memory item should maintain provenance information such as:

```text
Memory
├── ID
├── Content
├── Source
├── Source Device
├── Timestamp
├── Confidence
├── Trust Level
├── Created By
└── Version
```

External content is treated as **data**, not automatically as trusted instructions.

---

# 3. RAG

BAYMAX uses Retrieval-Augmented Generation to retrieve relevant information before generating an answer or planning an action.

The intended retrieval pipeline is:

```text
User Query
    │
    ▼
Query Processing
    │
    ├──────────────┐
    ▼              ▼
Lexical Search   Vector Search
    │              │
    └──────┬───────┘
           ▼
       Filtering
           │
           ▼
        Reranking
           │
           ▼
      Evidence Set
           │
           ▼
      Context Builder
           │
           ▼
       Local LLM
```

The system should preserve important exact information such as:

- File paths
- IDs
- Tool parameters
- Permissions
- Capability restrictions
- User instructions

Context compression must never silently remove security-critical information.

---

# 4. Agentic AI

BAYMAX is not intended to be simply a chatbot.

Its agent pipeline is:

```text
User
 │
 ▼
Router
 │
 ├── Direct Answer
 │
 ├── Retrieval
 │
 └── Agent Task
          │
          ▼
        Planner
          │
          ▼
   Structured Action
          │
          ▼
      Policy Gate
          │
          ▼
       Executor
          │
          ▼
      Verifier
          │
          ▼
   Canonical Event
          │
          ▼
        Memory
```

### Important security principle

The LLM **does not have authority**.

The LLM can propose:

```json
{
  "tool": "open_file",
  "arguments": {
    "path": "example.txt"
  }
}
```

But the deterministic security layer decides whether this action is permitted.

```text
LLM
 ↓
Tool Proposal
 ↓
Policy Engine
 ↓
Allowed?
 ├── NO → Reject
 └── YES
       ↓
   Execute
```

---

# 🔐 Capability-Based Security

BAYMAX uses capability-based authorization rather than giving the agent unrestricted access to the device.

Examples:

```text
READ_FILE
WRITE_FILE
OPEN_APPLICATION
SEND_NOTIFICATION
ACCESS_LOCATION
ACCESS_DEVICE_DATA
EXECUTE_TASK
```

Capabilities can have:

- Device scope
- Risk level
- Expiration
- User approval requirement
- Audit information

Example:

```text
Capability:
    WRITE_FILE

Device:
    Windows-PC

Scope:
    C:\Users\User\Documents\BAYMAX\*

Risk:
    Medium

Approval:
    Required
```

---

# 🌐 Networking Architecture

BAYMAX follows a **P2P-first architecture**.

Initial communication:

```text
Android
   │
   │ Secure P2P tunnel
   │
   ▼
Windows Core
```

The architecture should not depend entirely on mDNS.

Possible discovery methods:

1. QR-code pairing
2. Manual endpoint entry
3. Local discovery
4. Optional mDNS/DNS-SD
5. Future hybrid networking

### Why multiple discovery mechanisms?

Because local networks can have:

- AP isolation
- Multicast blocking
- Different subnets
- Different Wi-Fi bands
- Enterprise network restrictions

Therefore:

```text
mDNS
  ↓
if unavailable
  ↓
Manual / QR pairing
  ↓
Secure connection
```

Discovery is **not** treated as authentication.

---

# 🔒 Secure Communication

The intended transport architecture is:

```text
Application
     │
    gRPC
     │
     ▼
Secure Network Layer
     │
  WireGuard
     │
     ▼
Peer Device
```

WireGuard provides the encrypted network tunnel while gRPC provides structured application communication.

The application should still maintain:

- Device identity
- Peer authorization
- Capability authorization
- Connection state
- Audit information

---

# 🔄 Synchronization

BAYMAX uses an event-oriented synchronization model.

A simplified event:

```text
Event
├── event_id
├── device_id
├── sequence
├── entity_id
├── operation
├── payload
├── parent/version
└── metadata
```

The basic synchronization model:

```text
Android
   │
   │ Local Event
   ▼
Outbox
   │
   │ Sync
   ▼
Windows Core
   │
   │ Validate
   ▼
Canonical State
   │
   │ ACK
   ▼
Android
```

The Android device should only mark an event as synchronized after the Core has durably accepted it.

---

# 🧠 Canonical State

The Windows Core maintains canonical state for the MVP.

The system avoids relying on physical wall-clock timestamps for deterministic conflict resolution.

Instead, state transitions should use concepts such as:

- Event IDs
- Device IDs
- Sequence numbers
- Parent references
- Base versions
- Checkpoints

This prevents clock skew from becoming the primary ordering mechanism.

---

# 🔁 Task Ownership & Migration

Distributed task execution introduces a major problem:

```text
Device A
   │
   │ Task
   ▼
Device B
```

What happens if the connection breaks during migration?

BAYMAX uses:

- Task owner
- Lease
- `lease_epoch`
- Checkpoint
- Idempotency key
- Fencing

Conceptually:

```text
Task Owner = Device A
Lease Epoch = 5

Migration

Task Owner = Device B
Lease Epoch = 6
```

Any command from the stale epoch is rejected.

This prevents stale devices from continuing execution after ownership has moved.

---

# 📱 Android Architecture

The Android application is designed as an intermittently connected node.

It should **not** depend on a permanently running background WebSocket.

Architecture:

```text
Android App
│
├── UI
│
├── Local Database
│
├── Event Store
│
├── Encrypted Outbox
│
├── Sync Manager
│
├── Capability Adapters
│
└── WorkManager
```

Foreground sessions can use active connections while background synchronization can use Android-supported scheduling mechanisms.

The application must be designed around Android lifecycle restrictions.

---

# 💻 Windows Architecture

The Windows machine acts as the primary BAYMAX Core in the initial version.

```text
Windows BAYMAX Core
│
├── API / gRPC
├── Canonical State
├── Event Store
├── Task Engine
├── Policy Integration
├── AI Gateway
├── RAG
├── Persistent Memory
└── Audit Logs
```

The Core should remain usable even if:

- AI is unavailable
- RAG is unavailable
- Memory indexing fails
- Android is offline
- Hybrid networking is disabled

---

# 🧩 Modular Architecture

BAYMAX is intentionally modular.

External technologies should be accessed through adapters.

For example:

```text
MemoryService
      │
      ▼
MemoryRepository
      │
 ┌────┴─────┐
 ▼          ▼
Mem0      Custom DB
Adapter    Adapter
```

Similarly:

```text
ModelProvider
      │
 ┌────┼───────────┐
 ▼    ▼           ▼
Local  Ollama   Future Model
Model             Provider
```

This allows technologies to be replaced without rewriting BAYMAX Core.

---

# 🛠️ Technology Stack

## Android

- Kotlin
- Jetpack Compose
- Gradle
- Room / SQLite
- WorkManager
- gRPC client

Android development can be performed using **VS Code** with the Android SDK, JDK and Gradle configured locally.

---

## Windows / Backend

- Python or selected backend runtime
- SQLite
- gRPC
- Protocol Buffers
- Async processing where required

---

## Networking

- WireGuard
- gRPC
- Protobuf
- Optional mDNS/DNS-SD
- QR-based pairing
- Future Tailscale/Headscale evaluation

---

## AI

Potential components include:

- Local LLM runtime
- llama.cpp-compatible models
- Ollama-compatible model providers
- Structured tool calling
- Model routing

The final model should be selected based on benchmark results rather than assumptions.

---

## Memory

Possible technologies to evaluate:

- SQLite
- Vector index
- Mem0
- Letta-style memory architecture
- Custom provenance layer

External memory frameworks should be wrapped behind a BAYMAX memory interface.

---

## RAG

Potential components:

- SQLite FTS5
- Vector database/index
- Embedding model
- Metadata filtering
- Reranking

The project should benchmark whether a lightweight local implementation is sufficient before introducing additional infrastructure.

---

# 🗂️ Repository Structure

```text
baymax/
│
├── apps/
│   ├── android/
│   └── windows/
│
├── core/
│   ├── contracts/
│   ├── events/
│   ├── tasks/
│   ├── capabilities/
│   └── errors/
│
├── protocol/
│   ├── proto/
│   └── generated/
│
├── ai/
│   ├── gateway/
│   ├── router/
│   ├── rag/
│   ├── memory/
│   ├── provenance/
│   ├── context/
│   └── models/
│
├── network/
│   ├── wireguard/
│   ├── pairing/
│   ├── discovery/
│   └── grpc/
│
├── security/
│   ├── identity/
│   ├── policy/
│   ├── approvals/
│   └── audit/
│
├── adapters/
│   ├── tasker/
│   ├── macrodroid/
│   ├── syncthing/
│   └── external/
│
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── security/
│   ├── fuzz/
│   ├── e2e/
│   └── benchmarks/
│
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── protocols/
│   ├── stages/
│   ├── security/
│   └── troubleshooting/
│
├── configs/
├── scripts/
│
├── .github/
│   └── workflows/
│
├── README.md
├── CONTRIBUTING.md
└── CHANGELOG.md
```

---

# 👥 Development Team

BAYMAX is designed for a four-member team.

### Member 1 — Windows Core

Responsible for:

- Canonical state
- Event store
- SQLite
- Task engine
- Task ownership
- Checkpoints
- Windows services

### Member 2 — Android

Responsible for:

- Kotlin application
- Compose UI
- Local storage
- Outbox
- Synchronization client
- Android capabilities
- Android lifecycle handling

### Member 3 — AI

Responsible for:

- Local LLM
- Model gateway
- RAG
- Persistent memory
- Context management
- Agent orchestration
- AI benchmarking

### Member 4 — Network/Security/QA

Responsible for:

- WireGuard
- Pairing
- Discovery
- Capability authorization
- Security
- Fuzz testing
- Fault injection
- CI/CD
- Benchmarking

---

# 🏗️ Development Methodology

BAYMAX will be built incrementally.

```text
Stage 0
   ↓
Contracts
   ↓
Windows Core + Android Node
   ↓
Secure P2P
   ↓
Synchronization
   ↓
Canonical State
   ↓
Security
   ↓
Task Engine
   ↓
AI Gateway
   ↓
RAG
   ↓
Persistent Memory
   ↓
Agent Pipeline
   ↓
Cross-Device Continuity
   ↓
UI
   ↓
Reliability
   ↓
Hybrid Networking
```

Each stage must be independently verified before the next dependent stage begins.

---

# 🧪 Verification Philosophy

A feature is **not complete** simply because it works in the happy path.

Every major component must be tested against:

```text
Normal
   +
Duplicate
   +
Delayed
   +
Dropped
   +
Reordered
   +
Disconnected
   +
Restarted
   +
Unauthorized
   +
Corrupted
```

Examples:

### Synchronization

Test:

- Duplicate events
- Out-of-order events
- Dropped messages
- Network interruption
- Reconnection
- Partial synchronization

### Agent

Test:

- Invalid tool call
- Prompt injection
- Unauthorized capability
- Tool timeout
- Model failure
- Memory failure

### Task migration

Test:

- Owner crashes
- Network disconnects
- Stale owner executes
- Target accepts but source never receives ACK

---

# 📊 Benchmarking

BAYMAX should be compared against the project's chosen benchmark systems using measurable metrics.

Important metrics include:

### AI

- Time to First Token
- Total response latency
- Tool-call validity
- Tool argument accuracy
- Model memory consumption

### RAG

- Retrieval latency
- Retrieval hit rate
- Precision/recall where applicable
- Index size

### Synchronization

- Sync latency
- Convergence time
- Duplicate event rate
- Recovery time

### Agent

- Task completion rate
- Failure recovery rate
- Unauthorized action rejection rate

### Device

- CPU usage
- RAM usage
- Storage growth
- Android battery impact

---

# 🧪 Reliability Testing

BAYMAX includes deliberate fault injection.

Examples:

```text
Kill Windows Core
      ↓
Restart
      ↓
Recover canonical state
```

```text
Android
   ↓
Start Task
   ↓
Disconnect Network
   ↓
Windows continues/resumes
   ↓
Android reconnects
```

```text
LLM
 ↓
Invalid Tool Call
 ↓
Policy/Schema Validation
 ↓
Reject
 ↓
No Device Action
```

---

# 📚 Stage Documentation

Every stage must create:

```text
docs/stages/STAGE_XX_<NAME>.md
```

Each report must contain:

1. Objective
2. Scope
3. Architecture
4. Dependencies
5. Implementation
6. Files changed
7. API/protocol changes
8. Data model
9. Security considerations
10. Failure handling
11. Tests
12. Verification results
13. Performance measurements
14. Challenges
15. Root cause
16. Solution
17. Trade-offs
18. Known limitations
19. Reproduction instructions
20. Evidence
21. Interface for the next stage

---

# 🌿 Git Workflow

```text
main
 │
 └── develop
       │
       ├── feature/stage-01-contracts
       ├── feature/stage-02-windows-core
       ├── feature/stage-02-android
       ├── feature/stage-03-network
       └── ...
```

Workflow:

```text
Feature
   ↓
Local Tests
   ↓
Commit
   ↓
Pull Request
   ↓
Code Review
   ↓
CI
   ↓
Integration Tests
   ↓
develop
   ↓
Regression Tests
   ↓
Release Tag
   ↓
main
```

No direct pushes to `main`.

---

# 📦 What Should Be Committed?

Every completed stage should contain:

- Source code
- Tests
- Protocol definitions
- Database migrations
- Configuration examples
- Scripts
- Documentation
- Benchmark results
- Failure-test results
- Screenshots/logs where appropriate
- Changelog entry

Never commit:

- API keys
- Private keys
- Personal data
- Real user databases
- Passwords
- Production credentials

---

# 🛡️ Failure Isolation

A core BAYMAX principle is:

> **One failed component should not destroy the entire system.**

Examples:

```text
AI unavailable
      ↓
Basic local functionality continues
```

```text
RAG unavailable
      ↓
Direct answer/tool path can continue where safe
```

```text
Memory index corrupted
      ↓
Rebuild from canonical records
```

```text
Network unavailable
      ↓
Android continues offline
      ↓
Outbox waits for connection
```

```text
Hybrid relay unavailable
      ↓
P2P/local mode remains functional
```

---

# 🔮 Future Scope

Potential future extensions include:

- Additional operating systems
- Wearable integration
- Fitness-band integration
- IoT device integration
- Smart-home capabilities
- Tasker integration
- MacroDroid integration
- Syncthing-based data synchronization adapters
- Tailscale/Headscale hybrid networking
- Additional local models
- Multi-device agent execution
- More sophisticated memory systems
- Distributed task scheduling
- Privacy-preserving remote access

These are intentionally separated from the MVP.

---

# 🎓 Capstone Goal

The purpose of BAYMAX is not simply to build another chatbot.

The project demonstrates how to construct a:

> **local-first, persistent-memory, cross-device, agentic AI system with secure peer-to-peer communication and fault-tolerant synchronization.**

The major engineering challenge is integrating:

```text
AI
+
Persistent Memory
+
RAG
+
Agentic Execution
+
Distributed State
+
P2P Networking
+
Security
+
Android
+
Windows
```

while keeping the system modular, testable and replaceable.

---

# 📜 Project Status

> 🚧 **Active Development**

Current architecture:

```text
P2P First
   ↓
Local-First
   ↓
Android + Windows
   ↓
Persistent Memory
   ↓
Agentic AI
   ↓
Optional Hybrid Networking
```

---

# 📄 License

This project is licensed under the **MIT License**.

Copyright (c) 2026 BAYMAX Project Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

See the [LICENSE](LICENSE) file for the complete license text.

---

# 👨‍💻 Team

**BAYMAX — Capstone Project**

Built by a 4-members of undergraduate B.tech Computer Technology Student.

---

## ⭐ Engineering Principle

> **Build independently. Integrate through contracts. Verify before wiring. Document every failure. Keep dependencies replaceable.**

BAYMAX should not become a collection of tightly coupled features.

It should become a **modular system where every component can be tested, replaced, disabled and improved independently.**