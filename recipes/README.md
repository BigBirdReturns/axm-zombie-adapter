# Freeze recipes

A freeze recipe pins the last-known-good software stack for a GPU architecture
*before* the vendor drops it, so the hardware stays productive after driver
EOL. This directory is the project's compatibility memory: one file per
architecture, written while the stack is still easy to obtain, useful for
decades after it isn't.

Per DURABILITY.md threat T1: GPUs die of policy (driver EOL, CUDA gates)
before they die of physics. A recipe is the countermeasure — record everything
needed to rebuild the stack while the parts are still downloadable, and
archive the artifacts yourself.

## Recipe format

Each recipe records:

- **Architecture and cards** it covers.
- **Last full-support driver branch** and where support ended.
- **Last compatible CUDA / compute toolkit** version.
- **A pinned container base image** (tag AND digest — tags are mutable,
  digests are not).
- **Open-stack alternative** (Vulkan / NVK / ROCm status for the parts), since
  open drivers have no EOL policy.
- **Local archive checklist**: the installer files to mirror onto your own
  storage, with their sha256 sums as you verified them. Do not trust that a
  download URL will exist in ten years; it will not.
- **Date verified** and by whom.

## Rules

1. Verify every version claim against your own hardware before recording it.
   Recipes state what was tested, not what documentation promised.
2. Record digests and checksums at archive time. A recipe without checksums
   is a rumor.
3. Never delete a recipe for hardware that still exists somewhere. Storage is
   cheaper than rediscovery.

## Index

- `pascal.md` — NVIDIA Pascal (GTX 10xx, P40, P100). Template example; verify
  before relying on it.
