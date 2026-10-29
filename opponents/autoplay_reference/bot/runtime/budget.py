"""Central timing settings for live decisions, replay and simulator requests.

Edit the settings below to change processing ceilings. They include screen
preparation; callers can supply less time. Screen countdowns are never read.
"""
import time

# Inclusive processing ceilings, in seconds.
TURN_BUDGET = 25.0
SEND_BUDGET = 10.0
PREVIEW_BUDGET = 60.0
EMERGENCY_RESERVE_FRACTION = .20

# Time kept available for tablet taps, confirmations and retries.
EXECUTION_RESERVE = 12.0
PREVIEW_EXECUTION_RESERVE = 8.0

# The game's battle records (View Log pages, Battle Info cards), read inside the prompt's ceiling.
# A visit stops early enough to leave the decision its reserve; unread pages wait for the next prompt.
RECORDS_FIGHT_BUDGET = 10.0     # B34: one page 3.1 s, three pages 8.6 s, both cards included; after a catch-up page
                                # the cards may run on to the decision reserve (B37-B41: two pages took ~10 s)
RECORDS_SEND_BUDGET = 3.5       # B34 replacement visits: 2.7 s
RECORDS_DECISION_RESERVE = {"turn": 10.0, "send": 5.0}
DIALOG_READ_INTERVAL = .3       # an open prompt's answer dialogs (Choice restriction, no energy left, timeout)

# Simulator startup, shutdown and deadline transport overhead.
SIMULATOR_VERIFY_TIMEOUT = 10.0
SIMULATOR_SHUTDOWN_TIMEOUT = 3.0
SIMULATOR_DEADLINE_MARGIN = .1

# Menu navigation and human-readable progress (independent of battle clocks).
NAVIGATION_INITIAL_TIMEOUT = 30.0
NAVIGATION_STEP_TIMEOUT = 15.0
NAVIGATION_ATTEMPTS = 3
MATCHMAKING_TIMEOUT = 240.0
PROGRESS_INTERVAL = 5.0
CONSOLE_SLOW_OPERATION = 15.0
CLICK_RETRY_INTERVAL = 1.0
CLICK_ATTEMPTS = 3

# UI recovery cadence; separate from search ceilings and the observed game clocks.
UI_STALL_AFTER = 3.0
UI_DEEP_SCAN_INTERVAL = 2.0
UI_EVIDENCE_INTERVAL = 5.0
UI_UNKNOWN_NOTICE = 10.0
APP_RESUME_INTERVAL = 10.0


def ceiling(kind):
    """Read the phase's current setting at call time, including after imports."""
    return {"turn": TURN_BUDGET, "send": SEND_BUDGET, "preview": PREVIEW_BUDGET}[kind]


def allowance(kind, remaining=None, budget=None):
    default = ceiling(kind)
    value = min(default, max(0, default if budget is None else budget))
    return min(value, max(0, remaining - (EXECUTION_RESERVE if kind == "turn" else 0))) if remaining is not None else value


class Deadline:
    def __init__(self, timeout, epoch_ms=None):
        self.end = time.monotonic() + timeout
        if epoch_ms is not None:
            self.end = min(self.end, time.monotonic() + epoch_ms / 1000 - time.time())

    def remaining(self):
        return max(0, self.end - time.monotonic())

    def check(self):
        if not self.remaining():
            raise TimeoutError("deadline")
