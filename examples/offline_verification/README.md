# Offline release verification

An auditor working in an air-gapped environment must be able to decide whether
a published target has a complete TUF chain of trust, using only evidence that
was handed to them. This directory provides:

- `tuf/offline/` — a reusable offline verifier package and a
  `python -m tuf.offline` command line entry point;
- `simulated_repo.py` — an in-memory repository simulator (built on the
  test-suite simulator) that can publish both traditional `Metadata` and DSSE
  `SIMPLE` envelopes;
- `bundle_builder.py` — serializes simulator state into an evidence bundle;
- `generate_bundles.py` — emits one success bundle and several failure bundles,
  each changing exactly one piece of evidence.

## Trust model

The evidence bundle is only a transport container and is **not** a trust root.

- The sole trust anchor is the caller-supplied bootstrap root bytes.
- Every other role is accepted only after passing the same validation rules
  the online `tuf.ngclient.Updater` applies:
  root version chain → timestamp → snapshot → targets and delegated roles,
  including snapshot/timestamp hash pins and target length/hash checks.
- A role missing from the bundle ends verification as *insufficient evidence*.
  Nothing is read from any local cache or from the network.
- A role present in the bundle but not reachable through the verified chain is
  ignored: packaging bytes never grants them trust.
- Delegation lookup order and terminating-delegation semantics are identical to
  `Updater._preorder_depth_first_walk` (preorder, depth-first).
- Multiple root versions are allowed; conflicting bytes for the same role and
  version are rejected at bundle load with `BundleConflictError`.
- Signature inputs are the original packaged bytes: canonical JSON payloads for
  traditional metadata and PAE bytes for DSSE SIMPLE envelopes. Parsed and
  re-serialized bytes are never used as signature objects.
- The envelope mode (`metadata` or `simple`) is selected explicitly. A document
  in the other mode fails verification; there is no automatic fallback.

## Bundle layout

```
release-bundle/
  manifest.json
  metadata/
    1.root.json
    2.root.json
    timestamp.json
    snapshot.json
    targets.json
    vetted.json
  targets/
    vetted/release-1.0.tar.gz
```

`manifest.json` indexes raw bytes by role/version and target path:

```json
{
  "bootstrap_root": "metadata/1.root.json",
  "metadata": {
    "root": [
      {"version": 1, "path": "metadata/1.root.json"},
      {"version": 2, "path": "metadata/2.root.json"}
    ],
    "timestamp": [{"path": "metadata/timestamp.json"}],
    "vetted": [{"version": 1, "path": "metadata/vetted.json"}]
  },
  "targets": {
    "vetted/release-1.0.tar.gz": {
      "path": "targets/vetted/release-1.0.tar.gz"
    }
  }
}
```

Applications embedding the verifier can skip the filesystem entirely with
`EvidenceBundle.from_bytes(...)`.

## Verifying a target

```console
python -m tuf.offline \
    --bundle ./release-bundle \
    --target vetted/release-1.0.tar.gz \
    --at 2026-11-10T12:00:00Z \
    --envelope metadata \
    --output json
```

`--at` is an explicit, timezone-aware verification instant used for every
expiry check, so results are reproducible regardless of wall clock time.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | success: full chain of trust verified |
| 2 | bundle is malformed or contains conflicting bytes |
| 3 | verification failed (bad signature, hash, expiry, ...) |
| 4 | insufficient evidence: a required role or target is absent |
| 5 | target not referenced by any reachable targets role |

The report lists every role actually used, with its version, delegator, raw
file digests and signed-payload digests, plus the target's expected and
observed hashes. A third party can recompute every digest independently rather
than relying on a boolean verdict.

## Generating demo bundles

```console
python examples/offline_verification/generate_bundles.py ./offline-bundles
```

For each envelope type the generator writes seven bundles:

| Bundle | Single changed piece of evidence | Expected status |
|--------|----------------------------------|-----------------|
| `01-success` | none | success |
| `02-corrupt-target` | one target byte flipped | verification_failed |
| `03-tampered-timestamp` | one signature bit flipped, payload unchanged | verification_failed |
| `04-missing-delegation` | delegated role removed | insufficient_evidence |
| `05-missing-target` | target file bytes removed | insufficient_evidence |
| `06-conflicting-roots` | same root version shipped with different bytes | bundle error |
| `07-untrusted-extra-role` | target claimed only by an undelegated role | target_not_found |

## Programmatic API

```python
from datetime import UTC, datetime
from tuf.ngclient.config import EnvelopeType
from tuf.offline import EvidenceBundle, verify_target

bundle = EvidenceBundle.from_directory("./release-bundle")
result = verify_target(
    bundle,
    "vetted/release-1.0.tar.gz",
    datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC),
    EnvelopeType.METADATA,
)
assert result.success
for role in result.role_evidence:
    print(role.role, role.version, role.bytes_hashes["sha256"])
```
