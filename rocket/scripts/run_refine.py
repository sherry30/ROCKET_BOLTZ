import argparse

from ..refinement_config import RocketRefinmentConfig
from ..refinement_cryoem import run_cryoem_refinement
from ..refinement_xray import run_xray_refinement


def run_refinement(
    config: RocketRefinmentConfig | str,
    feats=None,
) -> RocketRefinmentConfig:
    """
    Dispatch to the appropriate refinement backend based on config.

    Parameters
    ----------
    config : RocketRefinmentConfig or str
        Run configuration.
    feats : dict or None
        Pre-computed Boltz-2 feats dict (batch dim = 1, CPU tensors).
        Required only when ``config.model == "boltz2"``; ignored otherwise.
        Prepare with ``rocket.prepare_boltz2_feats()``.
    """
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    model = getattr(config, "model", "alphafold")

    if model == "boltz2":
        from ..boltz2.refinement import run_boltz2_xray_refinement  # noqa: PLC0415
        if feats is None:
            feats_path = getattr(config.boltz2, "feats_path", None)
            if feats_path is None:
                raise ValueError(
                    "feats must be provided for model='boltz2' either via --feats "
                    "or boltz2.feats_path in the config. "
                    "Prepare them with rk.preprocess --model boltz2."
                )
            import pickle
            with open(feats_path, "rb") as fh:
                feats = pickle.load(fh)
        return run_boltz2_xray_refinement(config, feats)

    if config.datamode == "xray":
        return run_xray_refinement(config)
    elif config.datamode == "cryoem":
        return run_cryoem_refinement(config)


def cli_runrefine():
    parser = argparse.ArgumentParser(description="Run ROCKET refinement")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--feats",
        type=str,
        default=None,
        help="Path to a pickled Boltz-2 feats dict (required for model=boltz2)",
    )
    args = parser.parse_args()

    feats = None
    if args.feats is not None:
        import pickle
        with open(args.feats, "rb") as fh:
            feats = pickle.load(fh)

    run_refinement(args.config, feats=feats)
