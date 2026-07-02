# Freeze recipe: NVIDIA Pascal

> Status: TEMPLATE EXAMPLE. Version pins below reflect public support
> announcements as of early 2026 and must be verified on real hardware before
> this recipe is treated as authoritative. Fill in the checksum table from
> your own archive.

## Covers

- GeForce GTX 1050–1080 Ti, Titan X/Xp
- Tesla P40 (24 GB — the classic zombie-fleet card), P100
- Compute capability 6.x (sm_60, sm_61)

## Vendor stack (frozen)

- **Last full-support driver branch:** 580.x. NVIDIA moved Maxwell, Pascal,
  and Volta to legacy/maintenance after the 580 series (announced 2025);
  expect security-only updates on that branch, then nothing.
- **Last CUDA toolkit:** CUDA 12.x. CUDA 13 dropped compute capability < 7.5
  targets. Pin the newest 12.x your framework supports.
- **Container base:** an `nvidia/cuda:12.x` runtime image, pinned by digest
  (`nvidia/cuda@sha256:...`), paired with a host running the frozen 580
  driver. Record both.

## Open stack (no EOL)

- llama.cpp with the Vulkan backend runs on Pascal through Mesa/NVK or the
  frozen proprietary driver's Vulkan ICD; performance is lower than CUDA but
  the support horizon is unbounded. Prefer this path for post-EOL builds and
  keep the CUDA freeze for peak throughput.

## Local archive checklist

Mirror these to storage you control, then record sha256 sums here:

| Artifact | Version | sha256 | Archived |
|---|---|---|---|
| NVIDIA driver installer (.run) | 580.__ | _(fill in)_ | _(date)_ |
| CUDA 12.x toolkit installer | 12.__ | _(fill in)_ | _(date)_ |
| Container image (docker save) | nvidia/cuda@sha256:… | _(fill in)_ | _(date)_ |
| llama.cpp source snapshot | commit … | _(fill in)_ | _(date)_ |

## Verified

- _(date, person, hardware — fill in on first real verification)_
