"""Collection-only ownership: wait for real lift; ignore exhausted episode IDs."""


def recovery_execution_mask(proposed, previous, eligible, ready):
    # Keep the learner until a real lifted, opposed donor grip exists.
    # Once the expert has ownership, it retains it until the episode ends.
    # Ineligible reset environments must never trigger a new expert handoff.
    return (proposed | (previous & ~ready)) & eligible
