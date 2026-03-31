"""Optional Weights & Biases logger utilities.

This module is dependency-safe:
- If `wandb` is not installed, it degrades to a no-op logger.
- If disabled by config, all operations are no-ops.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class WandbLogger:
    """Minimal optional wrapper around Weights & Biases logging."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str | None,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None

        if not enabled:
            return

        if not WANDB_AVAILABLE:
            logger.warning(
                "W&B logging requested but wandb is not installed. "
                "Install with: pip install wandb"
            )
            return

        if not project:
            logger.warning(
                "W&B logging requested but no `wandb_project` was provided. "
                "Disabling W&B logging for this run."
            )
            self.enabled = False
            return

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
        )

        if self.run is not None:
            logger.info(f"W&B run initialized: {self.run.url}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        wandb.log(metrics, step=step)

    def log_molecule_3d(
        self,
        pdb_path: str | Path,
        *,
        name: str = "molecule_3d",
        step: int | None = None,
    ) -> None:
        if not self.enabled or self.run is None:
            return

        path = Path(pdb_path)
        if not path.exists():
            return

        try:
            wandb.log({name: wandb.Molecule(str(path))}, step=step)
        except Exception as exc:
            logger.debug(f"Failed to log 3D molecule {path}: {exc}")

    def log_structure_frame(
        self,
        coordinates: torch.Tensor | np.ndarray,
        *,
        topology_path: str | Path,
        name: str,
        step: int | None = None,
    ) -> None:
        if not self.enabled or self.run is None:
            return

        try:
            import mdtraj as md

            topology = md.load_pdb(str(topology_path)).topology
            if isinstance(coordinates, torch.Tensor):
                coordinates = coordinates.detach().cpu().numpy()

            coords_nm = np.asarray(coordinates).reshape(1, -1, 3) / 10.0
            trajectory = md.Trajectory(coords_nm, topology)

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                trajectory.save_pdb(str(tmp_path))
                self.log_molecule_3d(tmp_path, name=name, step=step)
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug(f"Failed to log live structure frame to W&B: {exc}")

    def log_trajectory_3d(
        self,
        trajectory_path: str | Path,
        *,
        max_frames: int = 100,
    ) -> None:
        if not self.enabled or self.run is None:
            return

        path = Path(trajectory_path)
        if not path.exists():
            return

        try:
            import mdtraj as md

            traj = md.load(str(path))
            n_frames = min(traj.n_frames, max_frames)
            html_path = path.parent / f"{path.stem}_animation.html"
            self._create_3dmol_animation(path, html_path, n_frames)
            wandb.log({f"trajectory_animation_{path.stem}": wandb.Html(str(html_path))})
        except Exception as exc:
            logger.debug(f"Failed to log W&B trajectory animation {path}: {exc}")

    def _create_3dmol_animation(
        self,
        pdb_path: Path,
        output_html: Path,
        n_frames: int,
    ) -> None:
        with open(pdb_path, encoding="utf-8") as file:
            pdb_content = file.read()

        pdb_content_js = (
            pdb_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        )

        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"    <title>Trajectory Animation - {pdb_path.stem}</title>",
            '    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>',
            "    <style>",
            (
                "        body { margin: 0; padding: 5px; font-family: Arial, "
                "sans-serif; background: white; }"
            ),
            (
                "        #container { width: 100%; max-width: 500px; height: "
                "350px; position: relative; margin: 0 auto; border: 1px solid "
                "#ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }"
            ),
            "        #controls { text-align: center; margin: 10px; }",
            (
                "        button { margin: 2px; padding: 6px 12px; font-size: "
                "12px; cursor: pointer; border: 1px solid #1f2937; background: "
                "#2f3b52; color: #ffffff; border-radius: 3px; }"
            ),
            "        button:hover { background: #1f2937; }",
            (
                "        #frameInfo { margin: 8px; font-size: 13px; "
                "font-weight: bold; text-align: center; }"
            ),
            "        h2 { text-align: center; margin: 8px 0; font-size: 16px; }",
            "        label { font-size: 12px; }",
            "        #speed { vertical-align: middle; width: 100px; }",
            "    </style>",
            "</head>",
            "<body>",
            f'    <h2 style="text-align: center;">Trajectory: {pdb_path.stem}</h2>',
            f'    <div id="frameInfo">Frame: 0 / {n_frames - 1}</div>',
            '    <div id="container"></div>',
            '    <div id="controls">',
            '        <button onclick="playAnimation()">▶ Play</button>',
            '        <button onclick="pauseAnimation()">⏸ Pause</button>',
            '        <button onclick="resetAnimation()">⏮ Reset</button>',
            '        <button onclick="prevFrame()">◀ Prev</button>',
            '        <button onclick="nextFrame()">▶ Next</button>',
            "        <label>Speed: ",
            (
                '            <input type="range" id="speed" min="100" '
                'max="2000" value="500" step="100">'
            ),
            '            <span id="speedLabel">500ms</span>',
            "        </label>",
            "    </div>",
            "    <script>",
            "        let viewer = null;",
            "        let currentFrame = 0;",
            "        let isPlaying = false;",
            "        let animationInterval = null;",
            "        let animationSpeed = 500;",
            f"        let numFrames = {n_frames};",
            (
                '        viewer = $3Dmol.createViewer("container", '
                '{backgroundColor: "white"});'
            ),
            f"        const pdbData = `{pdb_content_js}`;",
            '        viewer.addModelsAsFrames(pdbData, "pdb");',
            '        viewer.setStyle({}, {cartoon: {color: "spectrum"}});',
            "        viewer.zoomTo();",
            '        viewer.animate({loop: "forward", reps: 0});',
            "        viewer.stopAnimate();",
            "        viewer.render();",
            "        function updateFrameDisplay() {",
            "            const info = document.getElementById('frameInfo');",
            (
                "            info.textContent = `Frame: ${currentFrame} / "
                "${numFrames - 1}`;"
            ),
            "        }",
            "        function showFrame(frameNum) {",
            "            currentFrame = frameNum % numFrames;",
            "            viewer.setFrame(currentFrame);",
            "            viewer.render();",
            "            updateFrameDisplay();",
            "        }",
            "        function nextFrame() { showFrame(currentFrame + 1); }",
            "        function prevFrame() { showFrame(currentFrame - 1 + numFrames); }",
            "        function playAnimation() {",
            "            if (isPlaying) return;",
            "            isPlaying = true;",
            (
                "            animationInterval = setInterval(() => { "
                "nextFrame(); }, animationSpeed);"
            ),
            "        }",
            "        function pauseAnimation() {",
            "            isPlaying = false;",
            (
                "            if (animationInterval) { "
                "clearInterval(animationInterval); animationInterval = null; }"
            ),
            "        }",
            "        function resetAnimation() { pauseAnimation(); showFrame(0); }",
            (
                "        document.getElementById('speed').addEventListener("
                "'input', function(e) {"
            ),
            "            animationSpeed = parseInt(e.target.value);",
            (
                "            document.getElementById('speedLabel').textContent = "
                "animationSpeed + 'ms';"
            ),
            "            if (isPlaying) { pauseAnimation(); playAnimation(); }",
            "        });",
            "        updateFrameDisplay();",
            "    </script>",
            "</body>",
            "</html>",
        ]

        with open(output_html, "w", encoding="utf-8") as file:
            file.write("\n".join(html_lines))

    def log_artifact(
        self,
        file_path: str | Path,
        *,
        name: str | None = None,
        artifact_type: str = "artifact",
    ) -> None:
        if not self.enabled or self.run is None:
            return

        path = Path(file_path)
        if not path.exists():
            return

        artifact_name = name or path.stem
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        wandb.finish()
