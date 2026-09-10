# Security policy

The bundled aiohttp server is a reference inference service. It has request
validation and bounded trace/profile outputs, but it does not provide TLS,
authentication, authorization, rate limiting, or multi-tenant isolation. It
binds to loopback by default and must not be exposed directly to an untrusted
network.

Do not submit private datasets, model parameters, normalization files, secrets,
or machine-specific paths in issues or patches.
