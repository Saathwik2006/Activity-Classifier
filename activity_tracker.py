"""Turns noisy "what is happening right now" predictions into stable activities.

The model only sees about 1.6 seconds at a time. A short pause while typing (reading a paper,
thinking) looks like "background" to it. This module adds the rules a person would use:

  * switch_after (default 3 s): a new activity must show up for 3 s before it replaces the
    current one. One-second glitches are ignored.
  * end_after (default 10 s): an activity ends only after 10 s without seeing it. Short pauses
    stay part of the activity and do not stop the timer. The activity is recorded as ending at
    the last moment it was really seen, so the 10 s of waiting are not counted as work.
  * "unsure" / "warming up" count as "nothing seen".

Pure Python (no PyTorch or OpenCV needed), so it is easy to test: python test_activity_tracker.py
"""
from dataclasses import dataclass

NEUTRAL = ("unsure", "warming up")


@dataclass
class Segment:
    label: str
    start: float
    end: float

    @property
    def seconds(self):
        return self.end - self.start


class ActivityTracker:
    def __init__(self, background="background", switch_after=3.0, end_after=10.0, gap=1.5):
        self.background = background
        self.switch_after = switch_after
        self.end_after = end_after
        self.gap = gap                # a candidate may be missing for this long without being dropped
        self.current = background
        self.current_start = 0.0
        self.last_seen = 0.0          # last time the current activity was really seen
        self.cand = None              # [label, first_seen, last_seen] of a possible new activity
        self.segments = []            # finished work segments (background is not stored)

    # -- internal -------------------------------------------------------------------
    def _close(self, end):
        if self.current == self.background:
            return []
        end = max(end, self.current_start)
        seg = Segment(self.current, self.current_start, end)
        self.segments.append(seg)
        self.current = self.background
        return [("end", seg.label, seg.start, seg.end)]

    def _open(self, label, start):
        self.current, self.current_start = label, start
        return [("start", label, start, None)]

    # -- public ---------------------------------------------------------------------
    def update(self, t, raw):
        """t = time in seconds, raw = model label at this moment.
        Returns a list of events: ("start", label, start, None) or ("end", label, start, end).
        Times in events can be earlier than t (they are the real start/end of the activity)."""
        events = []
        if raw in NEUTRAL:
            raw = None
        if self.cand and t - self.cand[2] > self.gap:
            self.cand = None

        if raw is not None and raw != self.background and raw != self.current:
            if self.cand is None:
                self.cand = [raw, t, t]
            elif self.cand[0] == raw:
                self.cand[2] = t
            if self.cand[0] == raw and t - self.cand[1] >= self.switch_after:
                start = self.cand[1]
                events += self._close(start)
                events += self._open(raw, start)
                self.last_seen = t
                self.cand = None
        elif raw is not None and raw == self.current:
            self.last_seen = t

        if self.current != self.background and t - self.last_seen >= self.end_after:
            events += self._close(self.last_seen)
        return events

    def finish(self):
        """Call once when the video/camera stops. Closes the open activity."""
        return self._close(self.last_seen)

    def status(self, t):
        """(current activity, seconds since it started). Background gives 0 seconds."""
        if self.current == self.background:
            return self.current, 0.0
        return self.current, max(0.0, t - self.current_start)

    def totals(self):
        """{label: [total_seconds, number_of_segments]} including the open segment."""
        out = {}
        segs = list(self.segments)
        if self.current != self.background:
            segs.append(Segment(self.current, self.current_start, max(self.last_seen, self.current_start)))
        for s in segs:
            item = out.setdefault(s.label, [0.0, 0])
            item[0] += s.seconds
            item[1] += 1
        return out
