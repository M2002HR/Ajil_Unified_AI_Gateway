# Modules

This folder contains the three provider proxy projects used by `unified_gateway` as in-process modules:

- `gemini_proxy`
- `groq_proxy`
- `pollinations_proxy`

They are copied under `modules/` so the unified gateway can import their service classes directly.
If you initialize a git repository at root, you can replace these folders with real git submodules without changing gateway import paths.
