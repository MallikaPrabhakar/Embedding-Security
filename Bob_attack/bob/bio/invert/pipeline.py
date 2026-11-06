# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import logging
import os

import dask.bag
from bob.bio.base.pipelines.vanilla_biometrics import BioAlgorithmDaskWrapper
from bob.bio.base.pipelines.vanilla_biometrics import CSVScoreWriter
from bob.bio.base.pipelines.vanilla_biometrics import FourColumnsScoreWriter
from bob.bio.base.pipelines.vanilla_biometrics import checkpoint_vanilla_biometrics
from bob.bio.base.pipelines.vanilla_biometrics import dask_vanilla_biometrics
from bob.bio.base.pipelines.vanilla_biometrics import is_checkpointed
from bob.pipelines.utils import isinstance_nested, is_estimator_stateless
from dask.delayed import Delayed
from bob.pipelines.distributed import dask_get_partition_size
import bob.pipelines
import copy

logger = logging.getLogger(__name__)


def compute_scores(result, dask_client):
    if isinstance(result, Delayed) or isinstance(result, dask.bag.Bag):
        if dask_client is not None:
            result = result.compute(scheduler=dask_client)
        else:
            logger.warning("`dask_client` not set. Your pipeline will run locally")
            result = result.compute(scheduler="single-threaded")
    return result


def post_process_scores(pipeline, scores, path):
    written_scores = pipeline.write_scores(scores)
    return pipeline.post_process(written_scores, path)


from bob.pipelines.utils import flatten_samplesets

def get_inverted_references(biometric_references):
    inverted_references = copy.deepcopy(biometric_references)
    references = [r.reference_id for r in inverted_references]

    # breakdown sampleset
    inverted_references = flatten_samplesets(inverted_references)

    for sampleset in inverted_references:
        #sampleset.references = [sampleset.reference_id]
        sampleset.references = copy.deepcopy(references)
        for sample in sampleset:
            sample.key = sample.key + "-inverted"

        sampleset.key = sampleset.key + "-inverted"

    return inverted_references


def execute_inverted_vanilla_biometrics(
    pipeline,
    database,
    dask_client,
    groups,
    output,
    write_metadata_scores,
    checkpoint,
    dask_partition_size,
    dask_n_workers,
    **kwargs,
):
    """
    Function that executes the Vanilla Biometrics pipeline.

    This is called when using the ``bob bio pipelines vanilla-biometrics``
    command.

    This is also callable from a script without fear of interrupting the running
    Dask instance, allowing chaining multiple experiments while keeping the
    workers alive.

    Parameters
    ----------

    pipeline: Instance of :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`
        A constructed vanilla-biometrics pipeline.

    database: Instance of :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.abstract_class.Database`
        A database interface instance

    dask_client: instance of :py:class:`dask.distributed.Client` or ``None``
        A Dask client instance used to run the experiment in parallel on multiple
        machines, or locally.
        Basic configs can be found in ``bob.pipelines.config.distributed``.

    groups: list of str
        Groups of the dataset that will be requested from the database interface.

    output: str
        Path where the results and checkpoints will be saved to.

    write_metadata_scores: bool
        Use the CSVScoreWriter instead of the FourColumnScoreWriter when True.

    checkpoint: bool
        Whether checkpoint files will be created for every step of the pipelines.
    """

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    if write_metadata_scores:
        pipeline.score_writer = CSVScoreWriter(os.path.join(output, "./tmp"))
    else:
        pipeline.score_writer = FourColumnsScoreWriter(os.path.join(output, "./tmp"))

    # Check if it's already checkpointed
    if checkpoint and not is_checkpointed(pipeline):
        hash_fn = database.hash_fn if hasattr(database, "hash_fn") else None
        pipeline = checkpoint_vanilla_biometrics(pipeline, output, hash_fn=hash_fn)

        # TODO: look here Hatef

        # Here we have to checkpoint the `inversionAttack_transformer`
        # inversionAttack_transformer

        pipeline.inversionAttack_transformer = bob.pipelines.wrap(
            ["checkpoint"],
            pipeline.inversionAttack_transformer,
            features_dir=output,
            hash_fn=hash_fn,
        )

        pass

    # Load the background model samples only if the transformer requires fitting
    if all([is_estimator_stateless(step) for step in pipeline.transformer]):
        background_model_samples = []
    else:
        background_model_samples = database.background_model_samples()

    for group in groups:

        score_probes_file_name = os.path.join(
            output, f"scores-{group}" + (".csv" if write_metadata_scores else "")
        )
        score_inversionAttack_file_name = os.path.join(
            output,
            f"scores_inversion-{group}" + (".csv" if write_metadata_scores else ""),
        )

        biometric_references = database.references(group=group)
        probes = database.probes(group=group)
        inverted_references = get_inverted_references(biometric_references)

        # If there's no data to be processed, continue
        if len(biometric_references) == 0 or len(probes) == 0:
            logger.warning(
                f"Current dataset ({database}) does not have `{group}` set. The experiment will not be executed."
            )
            continue

        if dask_client is not None and not isinstance_nested(
            pipeline.biometric_algorithm, "biometric_algorithm", BioAlgorithmDaskWrapper
        ):
            # Scaling up
            if dask_n_workers is not None and not isinstance(dask_client, str):
                dask_client.cluster.scale(dask_n_workers)

            n_objects = max(
                len(background_model_samples), len(biometric_references), len(probes)
            )
            partition_size = None
            if not isinstance(dask_client, str):
                partition_size = dask_get_partition_size(dask_client.cluster, n_objects)
            if dask_partition_size is not None:
                partition_size = dask_partition_size

            pipeline = dask_vanilla_biometrics(pipeline, partition_size=partition_size,)

            # TODO: look here Hatef

            # Here we have to dask the `inversionAttack_transformer`
            # inversionAttack_transformer

            pipeline.inversionAttack_transformer = bob.pipelines.wrap(
                ["dask"],
                pipeline.inversionAttack_transformer,
                partition_size=partition_size,
            )

        logger.info(f"Running vanilla biometrics for group {group}")
        allow_scoring_with_all_biometric_references = (
            database.allow_scoring_with_all_biometric_references
            if hasattr(database, "allow_scoring_with_all_biometric_references")
            else False
        )
        scores_probes, scores_inversionAttack = pipeline(
            background_model_samples,
            biometric_references,
            probes,
            inverted_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        post_processed_scores = post_process_scores(
            pipeline, scores_probes, score_probes_file_name
        )
        _ = compute_scores(post_processed_scores, dask_client)

        post_processed_scores = post_process_scores(
            pipeline, scores_inversionAttack, score_inversionAttack_file_name
        )
        _ = compute_scores(post_processed_scores, dask_client)