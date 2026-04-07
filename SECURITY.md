# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

**Please DO NOT file a public issue** for security vulnerabilities.

Instead, send an email to **[contact.fardinsabid@gmail.com](mailto:contact.fardinsabid@gmail.com)** with:

- **Description** of the vulnerability
- **Steps to reproduce** (if applicable)
- **Potential impact** (what could an attacker do?)
- **Suggested fix** (if you have one)

### What to Expect

1. **Confirmation** within 48 hours that we received your report
2. **Investigation** and validation of the issue
3. **Fix development** (if confirmed)
4. **Coordinated disclosure** after fix is released

### Disclosure Policy

- We will release a patch as soon as possible
- We will credit the reporter (unless you wish to remain anonymous)
- We will publish a security advisory on GitHub

## Security Considerations for Aleam Users

### Cryptographic Security

Aleam provides **64 bits of true entropy per call** from the operating system's CSPRNG, combined with BLAKE2s cryptographic hashing. This provides:

- **Unpredictability**: Output cannot be predicted from previous values
- **Non-reproducibility**: No seeding, each run is unique
- **State-free**: No internal state to extract

### Known Limitations

| Issue | Mitigation |
|-------|------------|
| CPU speed | Use GPU acceleration (CuPy) |
| Platform dependence | Falls back gracefully |
| No reproducibility | Use Python's `random` if needed |

### Best Practices

1. **Do not use Aleam for security-critical systems** without additional entropy sources
2. **Do not rely on reproducibility** - Aleam is stateless by design
3. **Use GPU acceleration** for production workloads
4. **Keep dependencies updated** - regularly run `pip install --upgrade aleam`

## Cryptographic Dependencies

| Component | Purpose | Security Status |
|-----------|---------|-----------------|
| BLAKE2s | Cryptographic hash | ✅ Secure |
| getrandom() | Entropy (Linux) | ✅ Secure |
| BCryptGenRandom() | Entropy (Windows) | ✅ Secure |
| arc4random_buf() | Entropy (macOS) | ✅ Secure |

## Vulnerability Disclosure Timeline

| Date | Event |
|------|-------|
| 2026-03-30 | Initial release (v1.0.0) |
| 2026-04-06 | C++ migration (v1.0.3) |
| Future | Security updates as needed |

---

**Last Updated:** April 2026
