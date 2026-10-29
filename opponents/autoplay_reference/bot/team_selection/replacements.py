"""TS-09: deterministic hypothetical replacements, shared by every rollout phase."""


def choose(options, opponent, ranks, *, explicit=None):
    if explicit:
        selected = next((action for action in options if action["identity"] == explicit), None)
        if selected is None:
            raise ValueError("Requested replacement is unavailable")
        return selected
    if not options:
        raise ValueError("Unresolved replacement request")
    return max(options, key=lambda action: ranks.get(action["identity"], {}).get(opponent, 0) or 0)
