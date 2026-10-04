# Stage 0 — Repository Foundation

## Objective

Establish the development foundation for BAYMAX.

## Completed Components

- GitHub repository
- Branch strategy
- Branch protection
- Directory structure
- Protobuf contracts
- Core interfaces
- Error model
- Logging convention
- CI
- PR template
- Issue templates
- Coding conventions

## Branch Strategy

main → stable
develop → integration
feature/* → development

## Architecture Decisions

...

## Protobuf Contracts

...

## Core Interfaces

...

## Error Model

...

## Logging Convention

...

## CI Pipeline

...

## Verification

### Branch Protection

Test:
1. Attempt direct push to main.
2. Confirm rejection.

### Pull Request

1. Create feature branch.
2. Open PR.
3. Confirm CI runs.
4. Confirm review is required.
5. Confirm merge is blocked until requirements pass.

### CI

1. Push intentionally valid change.
2. Confirm CI passes.
3. Introduce controlled failure.
4. Confirm CI fails.

### Templates

Create test issue and PR.
Confirm correct templates appear.

## Problems Encountered

...

## Solutions

...

## Known Limitations

...

## Stage 1 Interface

...

## Evidence

Screenshots / logs / CI runs.