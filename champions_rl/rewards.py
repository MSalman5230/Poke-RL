"""Undiscounted match reward plus potential-based shaping (no damage farming)."""


def potential(obs):
    if obs["phase"] == "preview":
        return 0.
    own = sum(p["hp"] for p in obs["own"] if p["selected"])
    revealed = [p for p in obs["opponent"] if p["seen"]]
    enemy = sum(p["hp"] for p in revealed) + max(0,3-len(revealed))
    own_alive = sum(not p["fainted"] for p in obs["own"] if p["selected"])
    enemy_alive = 3-sum(p["fainted"] for p in revealed)
    return .5*(own-enemy)/3 + .5*(own_alive-enemy_alive)/3


def battle_reward(before, after, winner=None, scale=.15):
    terminal=after["ended"]
    outcome = (0. if winner is None else 1. if winner==after["side"] else -1.) if terminal else 0.
    # Terminal potential is zero, so summed shaping telescopes to a fixed
    # initial offset. A turn cap is truncation, not a manufactured win/loss.
    return outcome + scale*((0. if terminal else potential(after))-potential(before))
