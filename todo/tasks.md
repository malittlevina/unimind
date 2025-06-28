# Unimind TODO List

- [x] Populate persona logic files
- [x] Finalize tenets and ethics hooks
- [ ] Build model bridges  ← needs deeper runtime binding
- [ ] Extend actuator support  ← expand action vocab + integrations
- [ ] Link APIs to interfaces  ← complete wiring to real endpoints

### High Priority (Next Steps)
- [ ] Finish Codex memory integration ← enable querying and symbolic memory access
- [ ] Implement scroll reflection loop ← daily review of invoked scrolls + optimization
- [ ] Add emotion ↔ logic feedback layer ← amygdala ↔ prefrontal_cortex syncing
- [ ] Wire Unimind to main.py ← runtime boot logic should trigger full brain node load

### Unimind Infrastructure
- [ ] Create Unimind health monitor ← tracks active nodes, crash recovery, feedback
- [ ] Populate `foundation_manifest.json` fully ← add core values, constraints, permissions
- [ ] Generate internal event bus or signal routing system ← inter-node messaging
- [ ] Auto-load native models by brain node ← e.g. LLM to Broca’s/Wernicke’s, vision to occipital

### Integration + Expansion
- [ ] Add third-party toolchain hooks ← allow daemons to invoke external tools (e.g., Notion, GitHub)
- [ ] Enable Prom persona switching UI ← via `core_router.py` + `persona_manifest.json`
- [ ] Connect symbolic_reasoner to Codex ← to identify contradictions + suggest scrolls

### Interface & Deployment
- [ ] Build WebUI boot portal ← basic XR-friendly dashboard to interact with brain states
- [ ] Dockerize Unimind runtime ← package for easy deployment across systems
