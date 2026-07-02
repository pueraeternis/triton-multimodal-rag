# Security Policy

## Supported Scope

This repository is a **reference implementation** for demonstrating Triton BLS orchestration patterns. It is **not** a supported product and does not receive security patches on a schedule.

**Do not deploy this project as-is in production environments** without a full security review and hardening appropriate to your threat model.

---

## Reporting a Vulnerability

If you discover a security issue in this repository:

1. **Do not** open a public GitHub issue
2. Email the repository maintainer with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if any)

We will acknowledge receipt within a reasonable timeframe. Given the reference-implementation nature of this project, fixes may be best-effort.

---

## Known Security Limitations

This reference implementation intentionally omits production security controls:

| Control | Status |
|---------|--------|
| Authentication / authorization | Not implemented — Triton endpoints are open |
| Input validation | Minimal — client sends raw image tensors and text |
| Rate limiting | Not implemented |
| TLS / encryption in transit | Not configured — HTTP on port 8000 |
| Secrets management | `.env` file only — no vault integration |
| Network isolation | Docker bridge network — no firewall rules |
| Audit logging | Verbose Triton logs only — no structured audit trail |
| Dependency scanning | Not automated (planned in Plan 02) |

---

## Dependency Security

Third-party components (Triton, vLLM, Qdrant, HuggingFace models) carry their own security posture. Operators should:

- Pin container image versions in production
- Monitor upstream security advisories for Triton, vLLM, and PyTorch
- Restrict network access to Triton ports (8000, 8001, 8002) and Qdrant (6333)
- Use HuggingFace tokens with minimal scope

---

## Deployment Guidance

If you adapt this reference architecture for production:

1. Place Triton behind an authenticated API gateway
2. Enable TLS for all external communication
3. Restrict Qdrant to internal networks
4. Implement input size limits and content validation
5. Set up dependency vulnerability scanning in CI
6. Define an incident response process independent of this repository
