"""Explicit coverage for Showdown hydration; registry membership is not capture proof."""
BUILDER = "showdown-observed-state-v1"
FIELDS = {
    "Pokemon": ("identity", "set", "hp", "maxhp", "stats", "moves", "status", "progress", "types", "ability",
                "base_ability", "item", "mega_evolved", "hero_ready", "type_ability_used", "times_attacked", "transformed"),
    "Side": ("names", "pokemon", "boosts", "volatiles", "observation", "last_move", "mega_used", "fresh",
             "conditions", "replacement", "total_fainted", "fainted_last_turn", "fainted_this_turn"),
    "Field": ("weather", "weather_turns", "terrain", "terrain_turns", "trickroom", "extra"),
}
LIMITATIONS = {
    "transform": "Opposing Transform uses a snapshot of the known target; own copied UI slots remain unverified.",
    "unread_log_entry": "BE-22 retains last confirmed values for an unread or unknown View Log entry and logs possibly stale fields; no boosts inferred from intent.",
    "substitute_hp": "If remaining HP is unobserved, quarter of the known source maximum is an explicit assumption.",
    "hidden_durations": "Unobserved sleep/confusion/trapping durations use logged conditional hypotheses.",
    "pending_action": "An unobserved committed pivot reply uses the recorded greedy attack hypothesis.",
    "unobserved_hp": "Unrevealed opposing bench members are assumed full; active HP must be observed.",
    "received_hits": "Unresolved received-hit history blocks a current Rage Fist user; non-users retain an explicit unused zero assumption.",
}


def coverage():
    return {"state_builder": BUILDER, "fields": FIELDS, "limitations": LIMITATIONS,
            "backend": "pokemon-showdown", "capture_is_continuous": False}
