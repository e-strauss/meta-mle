# Diagrams

Requires [D2](https://d2lang.com/tour/install/) v0.9.0 or newer on PATH.
Uses the bundled TALA layout engine and built-in PNG renderer.

Edit `run-flow.d2`, then regenerate both checked-in images from the repository root:

```bash
bash docs/diagrams/render.sh
```

Preview the PNG in kitty:

```bash
kitten icat docs/diagrams/run-flow.png
```

The README embeds the SVG for sharp browser rendering. The PNG has an opaque
white background for consistent terminal display. Commit the source and both
exports together after editing the diagram.

The flow preserves the controller actions, exploration and expansion stages,
evaluation lock, candidate feedback, policy, memory and prompt connections.
Available policies and memories are listed inside their respective boxes.
