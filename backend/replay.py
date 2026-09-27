"""Single owned replay worker. Seeking rebuilds causal state, including pending forecasts."""
import threading
import time
import json
from pathlib import Path
import pandas as pd
from emulator.replay import events


class ReplayController:
    def __init__(self, data, split, dispatch, reset, advance, pending=None):
        self.data, self.split = data, split
        self.dispatch, self.reset, self.advance = dispatch, reset, advance
        self.pending = pending
        self.condition = threading.Condition(threading.RLock())
        self.mode = 'LIVE'
        self.status = 'idle'
        self.speed = 50
        if pending:
            self.start = min(e[0] for e in pending)
            self.end = max(e[0] for e in pending)+pd.Timedelta(minutes=15)
        else:
            try:
                manifest = json.loads((Path(data)/'manifest.json').read_text(encoding='utf-8'))
                self.start, self.end = pd.Timestamp(manifest['start']), pd.Timestamp(manifest['end'])
            except (OSError, ValueError, KeyError, TypeError):
                try:
                    points = pd.read_csv(Path(data)/'labels'/f'labels_{split}.csv',usecols=['T'])
                    times = pd.to_datetime(points['T'],format='mixed').dropna()
                    self.start, self.end = times.min(), times.max()+pd.Timedelta(minutes=15)
                    if pd.isna(self.start): raise ValueError('Empty replay dataset')
                except (OSError, ValueError, TypeError):
                    self.start = self.end = pd.Timestamp.now().floor('s')
        self.cursor = self.start
        self.index = 0
        self.target = None
        self.resume_after_seek = False
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self.run, daemon=True, name='transport-replay')
        self.thread.start()

    def state(self):
        with self.condition:
            return dict(mode=self.mode, status=self.status, speed=self.speed, timestamp=str(self.cursor),
                        start=str(self.start), end=str(self.end), error=self.error,
                        processed_events=self.index, total_events=len(self.pending or []))

    def control(self, action, speed=None, timestamp=None):
        with self.condition:
            if speed is not None:
                if speed not in (1, 10, 50): raise ValueError('Speed must be 1, 10 or 50')
                self.speed = speed
            if action == 'seek':
                target = pd.Timestamp(timestamp)
                if pd.isna(target) or target.tzinfo is not None or not self.start <= target <= self.end:
                    raise ValueError('Seek timestamp outside replay range')
                self.mode, self.target, self.status = 'REPLAY', target, 'seeking'
                self.resume_after_seek = False
            elif action == 'live':
                self.mode, self.status, self.target = 'LIVE', 'idle', None
                self.reset()
            elif action == 'play':
                if self.mode != 'REPLAY' or self.status in ('finished', 'error'):
                    self.mode, self.target, self.status = 'REPLAY', self.start, 'seeking'
                    self.resume_after_seek = True
                elif self.status != 'seeking': self.status = 'playing'
            elif action == 'pause':
                if self.status == 'playing': self.status = 'paused'
            elif action != 'speed': raise ValueError('Unknown replay action')
            self.condition.notify_all()
            return self.state()

    def run(self):
        previous = time.monotonic()
        while not self.closed:
            with self.condition:
                try:
                    if self.status == 'seeking' and self.target is not None:
                        # A command can replace the target between individual replay events.
                        destination = self.target
                        self.target = None
                        if self.pending is None:
                            self.pending = list(events(self.data, self.split, str(self.start-pd.Timedelta(minutes=20)), str(self.end)))
                        self.reset()
                        self.index = 0
                        self.cursor = self.start-pd.Timedelta(minutes=20)
                        self.destination = destination
                    if self.status == 'seeking':
                        if self.index < len(self.pending) and self.pending[self.index][0] <= self.destination:
                            at, _, path, payload = self.pending[self.index]
                            self.dispatch(path, payload)
                            self.cursor = at
                            self.index += 1
                        else:
                            self.cursor = self.destination
                            self.advance(self.cursor)
                            self.status = 'playing' if self.resume_after_seek else 'paused'
                    elif self.status == 'playing':
                        self.cursor = min(self.end, self.cursor+pd.Timedelta(seconds=min(time.monotonic()-previous, 1)*self.speed))
                        # Bounded batch keeps pause/seek responsive under high event density.
                        for _ in range(25):
                            if self.index >= len(self.pending) or self.pending[self.index][0] > self.cursor: break
                            _, _, path, payload = self.pending[self.index]
                            self.dispatch(path, payload)
                            self.index += 1
                        if self.index >= len(self.pending) or self.pending[self.index][0] > self.cursor:
                            self.advance(self.cursor)
                            if self.cursor >= self.end: self.status = 'finished'
                    previous = time.monotonic()
                except Exception as exc:
                    self.error, self.status = str(exc), 'error'
                self.condition.wait(timeout=.001 if self.status == 'seeking' else .03 if self.status == 'playing' else .2)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=10)
